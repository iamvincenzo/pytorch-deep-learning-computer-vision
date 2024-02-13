import torch
import torch.nn as nn
from math import sqrt


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, masked: bool) -> None:
        """
        Initialize the Scaled Dot-Product Attention module.

        Args:
            - head_dim (int): The dimensionality of the key and query vectors.
            - masked (bool): If True, applies masking to the attention scores.

        Returns:
            - None.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.head_dim = head_dim
        self.masked = masked
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:  
        """
        Compute scaled dot-product attention.

        Parameters:
            - queries (torch.Tensor): The queries tensor with shape (batch_size, seq_len_q, head_dim).
            - keys (torch.Tensor): The keys tensor with shape (batch_size, seq_len_k, head_dim).
            - values (torch.Tensor): The values tensor with shape (batch_size, seq_len_v, head_dim).
        
        Returns:
            - torch.Tensor: The output tensor with shape (batch_size, seq_len_q, head_dim).
        """
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / sqrt(self.head_dim)
        
        if self.masked:
            attention_scores = torch.tril(input=attention_scores)
            
        attention_weights = torch.softmax(input=attention_scores, dim=-1)
        
        output = torch.matmul(attention_weights, values)
        
        return output
         

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int) -> None:
        """
        Initialize the Multi-Head Attention module.
        
        Args:
            - num_heads (int): Number of attention heads.
            - embed_dim (int): Dimension of the input embeddings.
            
        Returns:
            - None.
        """
        super(MultiHeadAttention, self).__init__()        
        head_dim = embed_dim // num_heads
        
        self.query_projections = nn.ModuleDict(
            modules={f"query_projection_{i}" : nn.Linear(embed_dim, head_dim) for i in range(num_heads)}
        )
        self.key_projections = nn.ModuleDict(
            modules={f"key_projection_{i}" : nn.Linear(embed_dim, head_dim) for i in range(num_heads)}
        )
        self.value_projections = nn.ModuleDict(
            modules={f"value_projection_{i}" : nn.Linear(embed_dim, head_dim) for i in range(num_heads)}
        )
        
        self.scaled_dot_product = ScaledDotProductAttention(head_dim=head_dim, masked=False)
        
        self.output_projection = nn.Linear(in_features=num_heads * head_dim, out_features=embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention module.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            
        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        concatenated_heads = torch.tensor([])
        for query_projection, key_projection, value_projection in zip(self.query_projections.values(), self.key_projections.values(), self.value_projections.values()):
            projected_query = query_projection(x)
            projected_key = key_projection(x)
            projected_value = value_projection(x)
            
            attention_output = self.scaled_dot_product(projected_query, projected_key, projected_value)
            
            concatenated_heads = torch.cat(tensors=[concatenated_heads, attention_output], dim=-1)
            
        output = self.output_projection(concatenated_heads)
        
        return output
         

class Encoder(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int) -> None:
        """
        Initialize the Encoder module.
        
        Args:
            - num_heads (int): Number of attention heads in the Multi-Head Attention layer.
            - embed_dim (int): Dimension of the input embeddings.
            
        Returns:
            - None.
        """
        super(Encoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, embed_dim)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder module.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            
        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        attention_output = self.multi_head_attention(x)
        add_norm_1 = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(add_norm_1)
        add_norm_2 = self.layer_norm2(add_norm_1 + feed_forward_output)
        
        return add_norm_2


class Transformer(nn.Module):
    def __init__(self, num_encoder_blocks: int, num_heads: int, embed_dim: int, n_classes: int) -> None:
        """
        Initialize the Transformer model.
        
        Args:
            - num_encoder_blocks (int): Number of encoder blocks in the Transformer.
            - num_heads (int): Number of attention heads in each Multi-Head Attention layer.
            - embed_dim (int): Dimension of the input embeddings.
            
        Returns:
            - None.
        """
        super(Transformer, self).__init__()
        self.encoder_blocks = nn.ModuleDict(
            modules={f"encoder_block_{i}": Encoder(num_heads=num_heads, embed_dim=embed_dim) for i in range(num_encoder_blocks)}
        )
        
        self.output_linear = nn.Linear(in_features=embed_dim, out_features=n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            
        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        for encoder_block in self.encoder_blocks.values():
            x = encoder_block(x)
            
        x = self.output_linear(x)
            
        return x
