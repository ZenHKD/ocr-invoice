import torch
import torch.nn as nn
from model.rec.encoder import ViTEncoder
from model.rec.decoder import Decoder
from model.rec.tokenizer import Tokenizer
from model.rec.vocab import VOCAB

class ParSeq(nn.Module):
    def __init__(self, img_size=(128, 32), patch_size=(4, 8), embed_dim=384,
                 enc_depth=12, dec_depth=1, num_heads=6, charset=VOCAB, max_len=25):
        super(ParSeq, self).__init__()

        self.tokenizer = Tokenizer(charset)
        self.max_len = max_len

        self.encoder = ViTEncoder(img_size, patch_size, in_chans=3, embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads)
        self.decoder = Decoder(embed_dim, depth=dec_depth, num_heads=num_heads, max_len=max_len+2) # +2 for BOS/EOS

        self.text_embed = nn.Embedding(self.tokenizer.num_classes, embed_dim)
        self.head = nn.Linear(embed_dim, self.tokenizer.num_classes)

        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id

    def forward(self, images, target=None):
        # images: (B, C, H, W)
        # target: (B, T) - Token IDs (including BOS/EOS)

        # Encoder
        memory = self.encoder(images) # (B, S, C)

        if self.training and target is not None:
            # Training: Teacher forcing with AR/NAR mask (Simplified to AR for this implementation)
            # Remove EOS from input to decoder
            tgt_in = target[:, :-1]
            tgt_emb = self.text_embed(tgt_in)

            # Generate Permutation Mask (PLM) for ParSeq training
            mask = self.generate_permutation_mask(tgt_in.shape[1], images.device)

            out_dec = self.decoder(tgt_emb, memory, tgt_mask=mask)
            logits = self.head(out_dec)
            return logits

        else:
            # Inference: Autoregressive decoding
            B = images.shape[0]
            start_token = torch.full((B, 1), self.bos_id, dtype=torch.long, device=images.device)

            # Initial input: BOS
            tgt_in = start_token

            for _ in range(self.max_len + 1):
                tgt_emb = self.text_embed(tgt_in)

                # No mask needed as we build sequentially (or full attention on past)
                out_dec = self.decoder(tgt_emb, memory)

                logits = self.head(out_dec) # (B, T, V)
                last_logits = logits[:, -1, :] # (B, V)

                preds = last_logits.argmax(dim=-1, keepdim=True) # (B, 1)

                tgt_in = torch.cat([tgt_in, preds], dim=1)

                # Check if all batches have generated EOS (simplified check)
                if (preds == self.eos_id).all():
                    break

            return logits # Return full logits or decoded text in a real pipeline

    def generate_permutation_mask(self, seq_len, device):
        # Randomly choose between Causal Mask (50%) and Random Permutation (50%)
        r = torch.rand(1).item()
        
        if r < 0.5:
            # Standard AR (Left-to-Right): Future tokens are masked
            # Mask[i, j] = True means position i CANNOT see j
            mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        else:
            # Random Permutation: Shuffle context order
            # Simple approach: Allow seeing a random subset of context
            
            # Create random matrix
            rand_matrix = torch.rand((seq_len, seq_len), device=device)
            
            # Keep top k values (visible context)
            k = int(0.4 * seq_len) # Allow seeing 40% of random context
            topk_vals, _ = torch.topk(rand_matrix, k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            
            # Mask positions with values smaller than threshold (hide them)
            mask = rand_matrix < threshold
            
            # Ensure main diagonal is always open (self-attention)
            # Position i can always see itself to effectively aggregate context for prediction
            mask.fill_diagonal_(False)

        return mask

    @torch.inference_mode()
    def decode_greedy(self, images):
        # Wrapper for simple inference returning strings
        self.eval()
        B = images.shape[0]
        memory = self.encoder(images)

        tgt_in = torch.full((B, 1), self.bos_id, dtype=torch.long, device=images.device)

        # Store probability confidences if needed

        for _ in range(self.max_len + 1):
            tgt_emb = self.text_embed(tgt_in)
            out_dec = self.decoder(tgt_emb, memory)
            logits = self.head(out_dec)
            preds = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_in = torch.cat([tgt_in, preds], dim=1)

            if (tgt_in == self.eos_id).any(dim=1).all():
                break

        return self.tokenizer.decode(tgt_in)
