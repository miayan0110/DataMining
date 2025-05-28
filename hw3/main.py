import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
import argparse
import glob
import torch.nn.functional as F

from dataset import *
from torch.utils.tensorboard import SummaryWriter

class DINRecModel(nn.Module):
    def __init__(self,
                 emb_dim:    int   = 100,
                 gru_hidden: int   = 64,
                 att_hidden: int   = 64,
                 mlp_dims:  tuple  = (256,128),
                 dropout:  float  = 0.2):
        super().__init__()
        # 1) 双向 GRU，把 [B,M,D] -> [B,M,2*gru_hidden] -> 投影回 D
        self.gru = nn.GRU(
            input_size=emb_dim, 
            hidden_size=gru_hidden,
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.gru_proj = nn.Linear(2*gru_hidden, emb_dim)

        # 2) Additive MLP-Attention
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, att_hidden),
            nn.ReLU(),
            nn.Linear(att_hidden, 1)
        )

        # 3) Scoring MLP
        input_dim = emb_dim * 2
        layers = []
        for h in mlp_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.score_mlp = nn.Sequential(*layers)

    def forward(self,
                user_hist: torch.Tensor,
                imps_emb:  torch.Tensor,
                hist_mask: torch.Tensor) -> torch.Tensor:
        B, M, D = user_hist.size()
        _, n, _ = imps_emb.size()

        # --- 1) 序列编码：GRU + projection ---
        lengths = hist_mask.sum(1).cpu()
        packed = pack_padded_sequence(user_hist, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)  
        hist_enc, _ = pad_packed_sequence(out, batch_first=True, total_length=M)  # [B,M,2*H]
        hist_enc = self.gru_proj(hist_enc)  # [B,M,D]

        # --- 2) MLP-Attention pooling ---
        hist_exp = hist_enc.unsqueeze(2).expand(-1, -1, n, -1)   # [B,M,n,D]
        cand_exp = imps_emb.unsqueeze(1).expand(-1, M, -1, -1)   # [B,M,n,D]
        att_in   = torch.cat([hist_exp, cand_exp], dim=-1)      # [B,M,n,2D]
        att_s    = self.att_mlp(att_in).squeeze(-1)             # [B,M,n]
        att_s    = att_s.masked_fill(~hist_mask.unsqueeze(2), float('-inf'))
        att_w    = F.softmax(att_s, dim=1)                      # [B,M,n]
        pooled   = torch.einsum('bmn,bmd->bnd', att_w, hist_enc)  # [B,n,D]

        # --- 3) Scoring ---
        feat   = torch.cat([pooled, imps_emb], dim=-1)          # [B,n,2D]
        logits = self.score_mlp(feat).squeeze(-1)               # [B,n]
        return logits

def rec_collate_fn_din(batch):
    if len(batch[0]) == 4:
        ids, hist_list, imps_list, labels_list = zip(*batch)
    else:
        ids, hist_list, imps_list = zip(*batch)

    # 1) ids
    ids = torch.tensor(ids, dtype=torch.long)
    # 2) pad history 序列到同长度 M
    #    pad_sequence 会把 list of [m_i, D] -> [B, M, D]
    padded_hist = pad_sequence(hist_list, batch_first=True, padding_value=0.0)
    # 3) 构造 mask：真实位置 True，padding 位置 False
    masks = [torch.ones(h.shape[0], dtype=torch.bool) for h in hist_list]
    hist_mask = pad_sequence(masks, batch_first=True, padding_value=False)  # [B, M]
    # 4) stack impressions 和 labels
    imps   = torch.stack(imps_list,   dim=0)  # [B, n, D]

    if len(batch[0]) == 4:
        labels = torch.stack(labels_list, dim=0)  # [B, n]
        return ids, padded_hist, imps, labels, hist_mask
    else:
        return ids, padded_hist, imps, hist_mask

def train(model, optimizer, dataloader, args, loss_writer, auc_writer):
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    if args.resume:
        ckpt_path = sorted(glob.glob(f'{args.checkpoint_root}/*.pth'))[-1]
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, args.device)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss, all_probs, all_labels = 0.0, [], []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for ids, hist, imps, labels, masks in pbar:
            hist, imps, labels, masks = hist.to(args.device), imps.to(args.device), labels.to(args.device).float(), masks.to(args.device)
            optimizer.zero_grad()
            logits = model(hist, imps, masks)                      # [B, n]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.numel()
            probs = torch.sigmoid(logits).detach().cpu().view(-1)
            all_probs.append(probs)
            all_labels.append(labels.cpu().view(-1))

        # 平均 loss 與 AUC
        total_samples = len(dataloader.dataset) * imps.size(1)
        avg_loss = epoch_loss / total_samples
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        auc = roc_auc_score(all_labels, all_probs)

        pbar.set_postfix(loss=avg_loss, auc=auc)
        loss_writer.add_scalar('Loss', avg_loss, epoch+1)
        auc_writer.add_scalar('AUC', auc, epoch+1)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'{args.checkpoint_root}/epoch_{epoch+1:03d}.pth')

def validation(model, dataloader, args, auc_writer):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for ids, hist, imps, labels, masks in tqdm(dataloader, desc=f"Validating (history={args.max_history_len})"):
            hist   = hist.to(args.device)
            imps   = imps.to(args.device)
            labels = labels.to(args.device).float()
            masks  = masks.to(args.device)

            logits = model(hist, imps, masks)
            probs  = torch.sigmoid(logits).cpu().view(-1)
            all_probs.append(probs)
            all_labels.append(labels.cpu().view(-1))

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Max History Len = {args.max_history_len}, AUC = {auc:.4f}")
    auc_writer.add_scalar('AUC_vs_HistoryLen', auc, args.max_history_len)

def evaluate(model, dataloader, device, output_csv_path):
    model.to(device).eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for ids, hist, imps, mask in tqdm(dataloader, desc="Predicting"):
            hist = hist.to(device)    # [B, M, D]
            imps = imps.to(device)    # [B, n, D]
            mask = mask.to(device)    # [B, M]

            logits = model(hist, imps, mask)          # [B, n]
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, n]

            all_ids.extend([int(i) for i in ids.tolist()])
            all_probs.append(probs)

    # 合并成 [N, n] 矩阵
    probs_matrix = np.vstack(all_probs)
    n = probs_matrix.shape[1]

    columns = ['id'] + [f'p{i}' for i in range(1, n+1)]
    df = pd.DataFrame(
        data = np.column_stack([all_ids, probs_matrix]),
        columns = columns
    )
    df['id'] = df['id'].astype(int)
    for c in columns[1:]:
        df[c] = df[c].astype(float)

    df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"Saved probability predictions to {output_csv_path}")
    return df


def load_checkpoint(path, model, optimizer=None, device='cuda:0'):
    print(f'loading checkpoint from {path}...')
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return start_epoch

def get_args():
    parser = argparse.ArgumentParser(description="News Recommendation Training")

    # Model hyperparameters
    parser.add_argument("--emb_dim", type=int, default=100,
                        help="Dimensionality of entity and news embeddings")
    parser.add_argument("--gru_hidden", type=int, default=64,
                        help="Hidden dimension size in the gru attention MLP")
    parser.add_argument("--att_hidden", type=int, default=64,
                        help="Hidden dimension size in the attention MLP")
    parser.add_argument("--mlp_dims", type=int, nargs="+", default=[256, 128],
                        help="List of hidden layer sizes for the scoring MLP (e.g. --mlp_dims 256 128)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability used throughout the model")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_root", type=str, default="./ckpt/dinrec_v3",
                        help="Root to save/load model checkpoint")
    parser.add_argument("--logdir", type=str, default="./records",
                        help="Path to save logs")
    parser.add_argument("--output_csv_path", type=str, default="result.csv",
                        help="Path to save csv")
    
    # Experiment hyperparameters
    parser.add_argument("--max_history_len", type=int, default=0,
                        help="Max length of user history using for prediction, 0 means no limitation")
    
    # Device
    parser.add_argument("--device", type=str,
                        default="cuda:3" if torch.cuda.is_available() else "cpu",
                        help="Device (cpu or cuda)")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    loss_subdir = f'{args.logdir}/dinrec_v3_loss'
    auc_subdir = f'{args.logdir}/dinrec_v3_auc'
    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(loss_subdir, exist_ok=True)
    os.makedirs(auc_subdir, exist_ok=True)
    
    loss_writer = SummaryWriter(loss_subdir)
    auc_writer = SummaryWriter(auc_subdir)

    train_loader = DataLoader(TrainingData(), batch_size=args.batch_size, shuffle=True, collate_fn=rec_collate_fn_din, num_workers=4, pin_memory=True)

    model = DINRecModel(args.emb_dim, args.gru_hidden, args.att_hidden, args.mlp_dims, args.dropout)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_loader, args, loss_writer, auc_writer)

    ckpt_path = sorted(glob.glob(f'{args.checkpoint_root}/*.pth'))[-1]
    start_epoch = load_checkpoint(ckpt_path, model, optimizer, args.device)

    # # experiment of max_history_len
    # valid_subdir = f'{args.logdir}/valid_auc'
    # os.makedirs(valid_subdir, exist_ok=True)
    # valid_writer = SummaryWriter(auc_subdir)
    # for i in range(0, 11, 5):
    #     args.max_history_len = i
    #     validation(model, train_loader, args, valid_writer)

    test_loader = DataLoader(TestData(), batch_size=1, shuffle=False, collate_fn=rec_collate_fn_din, num_workers=4, pin_memory=True)
    evaluate(model, test_loader, args.device, args.output_csv_path)


# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7
