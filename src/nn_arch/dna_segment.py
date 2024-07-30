"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Any

"""
class-0: Background class
class-1: Peanut Seed
"""

"""
Which metric to focus on:
[1] Dice co-efficient: 
    
    -   It measures the overlap between the predicted and groundtruth regions. it is particularly useful
        for binary and multi-class segmentation tasks and is sensitive to the presence of small objects. 

    -   High Dice co-efficient values indicate good segmentation performance.

    -   You should focus on the Dice coefficient for each class separately to ensure that the model
        performs well across all classes, especially in cases where class imbalance is a concern.

[2] mean Intersection over Union (IoU):

    -   It measures the average overlap between the predicted and groundtruth regions across all classes.
        It is a more stringent (stiff, rigid) metric than the Dice co-efficient because it penalizes
        both false positives and false negatives more heavily.
    
    -   High mean Intersection over Union values indicate that the model segments the images accurately
        across all classes.

    -   You should focus on the mean Intersection over Union to get an overall sense of the model's
        segmentation performance across all classes.

Class-Specific performance:
    -   If you are particularly concerned about the performance on individual classes (e.g. due to 
        class imbalance or specific importance of certain classes), focus on the Dice co-efficient
        for each class. This will help you identify any classes where the model might be 
        underperforming.

    -   If you want an overall measure of segmentation quality that takes into account the performance
        across all classes, focus on the mean IoU. This will give you a comprehensive view of how well
        the model segments the images on average.
"""

# Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self,alpha=1.0,gamma=2.0,reduction='mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha # assign alpha value
        self.gamma = gamma # assign gamma value
        self.reduction = reduction # assign reduction value; 'mean', 'none', 'sum'

    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none') # cross-entropy loss
        p_t = torch.exp(-ce_loss) # probability of the positive class
        focal_loss = self.alpha * (1-p_t) ** self.gamma * ce_loss # calculate the focal loss

        if self.reduction=='mean': # the losses are averaged over the batch
            return focal_loss.mean()
        elif self.reduction=='sum': # the losses are summed over the batch
            return focal_loss.sum()
        else: # the loss will return as-is for each element in the batch; no reduction will be applied
            return focal_loss

# V-Net: Fully CNN for Volumetric Medical Image Segmentation: https://arxiv.org/abs/1606.04797
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1) -> None:
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth # assign smooth param value

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions) # apply sigmoid to predictions to get probabilities
        predictions = predictions.view(-1) # flatten the predictions from (N, H, W) to (N, H*W)
        targets = targets.view(-1) # flatten the targets from (N, H, W) to (N, H*W)
        intersection = (predictions * targets).sum() # calculate intersections
        union = predictions.sum() + targets.sum() # calculate union
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth) # compute dice co-efficient
        dice_loss = 1 - dice_coeff # compute dice loss
        return dice_loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5) -> None:
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss() # initialize the focal loss
        self.dice_loss = BinaryDiceLoss() # initialize the dice loss
        self.alpha = alpha # set value of alpha
        self.beta = beta # set value of beta
        """
        Adjust alpha and beta based on the performance:
        [1] Class imbalance: Increase alpha to give more weight to Focal Loss, which handles
            class imbalance well.
        [2] Overall Segmentation Accuracy: Increase beta to give more weight to Dice Loss/ Binary Dice Loss
            which measures overlap directly.
        """

    def forward(self, predictions, targets):
        focal_loss = self.focal_loss(predictions,targets) # calculate the focal loss
        dice_loss = self.dice_loss(predictions,targets) # calculate the dice loss
        combined_loss = self.alpha * focal_loss + self.beta * dice_loss # calculate the linear combination of focal loss and dice loss
        return combined_loss # return combined loss

class PatchEmbedding(nn.Module):
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size # initialize patch size
        self.num_patches_h = img_height // patch_size # calculate the number of patches based on the image height
        self.num_patches_w = img_width // patch_size # calculate the number of patches based on the image width
        self.num_patches = self.num_patches_h * self.num_patches_w # calculate the total number of patches
        self.embed_dim = embed_dim # initialize embedding dimension
        
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim) # linear layer to project each patch to embedding dimension
        
    def forward(self, x):
        B, C, H, W = x.shape # get the batch size, channels, height and width of the input
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # unfold the input tensor to extract patches
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size) # rearrange the patches to be in the contiguous block of memory
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.patch_size * self.patch_size * C) # permute the dimension to bring the patch dimension to the front and flatten the patch into single vector
        x = self.proj(x) # project the flattened patches to the embedding dimension
        return x # return the embedded patches

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # initialize position embeddings as a learnable parameter
        
    def forward(self, x):
        return x + self.position_embeddings # add position embeddings to the input tensor X

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads # initialize the number of heads
        self.head_dim = embed_dim // num_heads # set dimension of the each attention head
        self.scale = self.head_dim ** -0.5 # scaling factor for attention scores
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3) # linear layer to project input to queries, key, and values
        self.fc = nn.Linear(embed_dim, embed_dim) # linear layer to project the concatenated outputs of attention heads
        
    def forward(self, x):
        B, N, D = x.shape # get the batch size, number of patches, and embedding dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) # project input to queries, keys, and values and reshape for multi-head attention
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3, dim=0) # split the queries, keys, and values for each head
        
        q = q.squeeze(0) # remove the redundant dimension for queries
        k = k.squeeze(0) # remove the redundant dimension for keys
        v = v.squeeze(0) # remove the redundant dimension for values
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # compute the attention scores
        attn = attn.softmax(dim=-1) # apply softmax to get attention weights
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D) # compute the output by applying attention weights to the values
        out = self.fc(out) # project the concatenated outputs of the attention heads
        return out # return the output of the multi-head self-attention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim) # layer normalization applied before multi-head self-attention
        self.norm2 = nn.LayerNorm(embed_dim) # layer normalization applied before the MLP
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads) # multi-head self-attention layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), # first linear layer projects from embed_dim to mlp_dim
            nn.GELU(), # GELU activation function
            nn.Linear(mlp_dim, embed_dim), # second linear layer projects back to embed_dim
        ) 
        
    def forward(self, x):
        x = x + self.mhsa(self.norm1(x)) # apply layer normalization, then multi-head self-attention, and add the result to the input (Residual Connection)
        x = x + self.mlp(self.norm2(x)) # apply layer normalization, then MLP, and add the result to the input (Residual Connection)
        return x # return the output

class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim):
        super(VisionTransformerEncoder, self).__init__()
        self.patch_embedding = PatchEmbedding(img_height, img_width, patch_size, in_channels, embed_dim) # initialize patch embedding
        self.positional_encoding = PositionalEncoding(self.patch_embedding.num_patches, embed_dim) # initialize positional embedding
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ]) # initialize a list of transformer encoder layers
        
    def forward(self, x):
        x = self.patch_embedding(x) # apply patch embedding to input
        x = self.positional_encoding(x) # add positional encoding to the embeddings
        skip_connections = [] # initialize a list to store skip connections
        for layer in self.encoder_layers:
            x = layer(x) # apply each transformer encoder layer
            skip_connections.append(x) # store the output for skip connections
        return x, skip_connections # return the final output and skip connections

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim) # layer normalization before self-attention
        self.norm2 = nn.LayerNorm(embed_dim) # layer normalization before cross-attention
        self.norm3 = nn.LayerNorm(embed_dim) # layer normalization before MLP
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads) # self-attention layer
        self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads) # cross-attention layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), # first linear layer of MLP
            nn.GELU(), # GELU activation function
            nn.Linear(mlp_dim, embed_dim), # second linear layer of MLP
        )
        
    def forward(self, x, encoder_output):
        x = x + self.self_attn(self.norm1(x)) # apply self-attention and add residual connection
        x = x + self.cross_attn(self.norm2(x), encoder_output) # applt cross-attention and add residual connection
        x = x + self.mlp(self.norm3(x)) # apply MLP and add residual connection
        return x # return the output of the decoder layer

class VisionTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_dim):
        super(VisionTransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ]) # initialize a list of transformer decoder layers
        
    def forward(self, x, encoder_output):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output) # apply each transformer decoder layer
        return x # return the final output

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # initialize the transposed convolution layer for upsampling
        
    def forward(self, x):
        return self.upconv(x) # apply the transposed convolution to upsample the input

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # initialize the first convolution layer with kernel size 3 and padding 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # initialize the second convolution layer with kernel size 3 and padding 1
        
    def forward(self, x):
        x = F.relu(self.conv1(x)) # apply the first convolution layer followed by ReLU activation
        x = F.relu(self.conv2(x)) # applt the second convolution layer followed by ReLU activation
        return x # return the output

class DNASegmentModel(pl.LightningModule):
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, optimizer):
        super(DNASegmentModel, self).__init__()

        self.optimizer = optimizer # set optimizer

        # stage-1: encoder, bottleneck, and decoder
        self.stage1_encoder = VisionTransformerEncoder(img_height, img_width, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim)
        self.stage1_bottleneck = ConvBlock(embed_dim, embed_dim * 2)
        self.stage1_decoder = VisionTransformerDecoder(embed_dim, num_layers, num_heads, mlp_dim)
        
        # stage-2: encoder, bottleneck, and decoder
        self.stage2_encoder = VisionTransformerEncoder(img_height, img_width, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim)
        self.stage2_bottleneck = ConvBlock(embed_dim, embed_dim * 2)
        self.stage2_decoder = VisionTransformerDecoder(embed_dim, num_layers, num_heads, mlp_dim)
        
        # upsampling blocks for stage 1
        self.upconv_blocks_stage1 = nn.ModuleList([
            UpsampleBlock(embed_dim * 2, embed_dim),
            UpsampleBlock(embed_dim, embed_dim // 2),
            UpsampleBlock(embed_dim // 2, embed_dim // 4),
            UpsampleBlock(embed_dim // 4, embed_dim // 8),
            UpsampleBlock(embed_dim // 8, embed_dim // 16),
            UpsampleBlock(embed_dim // 16, embed_dim // 32),
        ])
        
        # upsampling blocks for stage 2
        self.upconv_blocks_stage2 = nn.ModuleList([
            UpsampleBlock(embed_dim * 2, embed_dim),
            UpsampleBlock(embed_dim, embed_dim // 2),
            UpsampleBlock(embed_dim // 2, embed_dim // 4),
            UpsampleBlock(embed_dim // 4, embed_dim // 8),
            UpsampleBlock(embed_dim // 8, embed_dim // 16),
            UpsampleBlock(embed_dim // 16, embed_dim // 32),
        ])
        
        # final convolution to get the desired number of classes
        self.final_conv = nn.Conv2d(embed_dim // 32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # stage 1
        x1, skip_connections1 = self.stage1_encoder(x)
        x1 = self.stage1_bottleneck(x1)
        x1 = self.stage1_decoder(x1, skip_connections1[-1])
        
        # upsample and concatenate skip connections of stage 1
        for i, upconv_block in enumerate(self.upconv_blocks_stage1):
            x1 = upconv_block(x1)
            if i < len(skip_connections1):
                x1 = torch.cat((x1, skip_connections1[-(i+1)]), dim=1)
        
        # concatenate original input image with stage 1 output
        x2_input = torch.cat((x, x1), dim=1)
        
        # stage 2
        x2, skip_connections2 = self.stage2_encoder(x2_input)
        x2 = self.stage2_bottleneck(x2)
        
        # add connections from stage 1 decoder to stage 2 encoder
        for i in range(len(skip_connections1)):
            x2 = x2 + skip_connections1[i]
        
        x2 = self.stage2_decoder(x2, skip_connections2[-1])
        
        # upsample and concatenate skip connections of stage 2
        for i, upconv_block in enumerate(self.upconv_blocks_stage2):
            x2 = upconv_block(x2)
            if i < len(skip_connections2):
                x2 = torch.cat((x2, skip_connections2[-(i+1)]), dim=1)
        
        x2 = self.final_conv(x2)
        return x2
    
    def training_step(self, batch, batch_idx) -> None:
        
        images, masks = batch # load input images and input masks from single-single batch
        outputs = self(images) # calculate the prediction
        
        # compute metrics
        preds = torch.argmax(outputs, dim=1) # convert raw outputs to predicted class labels
        
        combined_loss = CombinedLoss(alpha=0.5, beta=0.5) # initialize the combined loss
        loss = combined_loss(outputs,masks) # calculate the focal loss; attention to the below comment about params

        """
        Here in the loss calculation I have passed 'outputs' instead of the 'preds' because
        cross-entropy loss function needs the raw logits to compute the probabilities for each
        class and then calculate the loss. The cross-entropy function internally applies the
        softmax operation to the logits to compute the probabilities and cross-entropy loss.
        """

        mean_iou_score = self.mean_iou(preds,masks) # calculate the mean iou score over all the classes
        """
        Here in the mean_iou function, "preds" parameter is passed because here there is need of
        predicted label classes, not need of the raw logits.
        """

        dice_coeff_bg, dice_coeff_peanut_seed = self.dice_coefficient(preds,masks) # calculate the dice_coefficient separately for all avaiilable classes

        # log metrics
        self.log('DNASegment_train_combined_loss',loss,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the loss logs for visualization
        self.log('DNASegment_train_mean_IoU',mean_iou_score,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the mean IoU logs for visualization
        self.log('DNASegment_train_dice_coeff_bg',dice_coeff_bg,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the dice_coeff_bg logs for visualization
        self.log('DNASegment_train_dice_coeff_peanut_seed',dice_coeff_peanut_seed,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the dice_coeff_peanut_seed logs for visualization

        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        
        images, masks = batch # load input images and input masks from single-single batch
        outputs = self(images) # calculate the prediction
        
        # compute metrics
        preds = torch.argmax(outputs, dim=1) # convert raw outputs to predicted class labels
        
        combined_loss = CombinedLoss(alpha=0.5, beta=0.5) # initialize the combined loss
        loss = combined_loss(outputs,masks) # calculate the focal loss; attention to the below comment about params

        """
        Here in the loss calculation I have passed 'outputs' instead of the 'preds' because
        cross-entropy loss function needs the raw logits to compute the probabilities for each
        class and then calculate the loss. The cross-entropy function internally applies the
        softmax operation to the logits to compute the probabilities and cross-entropy loss.
        """

        mean_iou_score = self.mean_iou(preds,masks) # calculate the mean iou score over all the classes
        """
        Here in the mean_iou function, "preds" parameter is passed because here there is need of
        predicted label classes, not need of the raw logits.
        """

        dice_coeff_bg, dice_coeff_peanut_seed = self.dice_coefficient(preds,masks) # calculate the dice_coefficient separately for all avaiilable classes

        # log metrics
        self.log('DNASegment_validation_combined_loss',loss,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the loss logs for visualization
        self.log('DNASegment_validation_mean_IoU',mean_iou_score,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the mean IoU logs for visualization
        self.log('DNASegment_validation_dice_coeff_bg',dice_coeff_bg,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the dice_coeff_bg logs for visualization
        self.log('DNASegment_validation_dice_coeff_peanut_seed',dice_coeff_peanut_seed,on_step=True,on_epoch=True,prog_bar=True,enable_graph=True) # save the dice_coeff_peanut_seed logs for visualization

        return loss
    
    def mean_iou(self, predictions, targets, num_classes=2) -> Any:
        ious = [] # create an empty list to store class-wise ious
        predictions = predictions.view(-1) # flatten the predictions from (N, H, W) to (N, H*W)
        targets = targets.view(-1) # flatten the targets from (N, H, W) to (N, H*W)

        for cls in range(num_classes): # iterate over each class
            predictions_inds = predictions == cls # create a binary mask for the current class in the predictions
            target_inds = targets == cls # create a binary mask for the current class in the targets
            intersection = (predictions_inds & target_inds).sum().item() # calculate the intersection for the current class
            union = (predictions_inds | target_inds).sum().item() # calculate the union for the current class

            if union == 0:
                ious.append(float('nan')) # if there is no groundtruth, do not include this in IoU calculation
            else:
                iou = intersection / union # calculate the IoU for the current class
                ious.append(iou) # append the IoU of the cuurent class to the list

        mean_iou = torch.tensor(ious).mean() # return the mean iou over all classes

        return mean_iou.item() # return a python scalar instead of torch tensor
    
    def dice_coefficient(self, predictions, targets, num_classes=2, smooth=1e-5) -> Any:
        dice_scores = [] # define empty list to store class-wise dice scores

        for cls in range(num_classes):
            predictions = (predictions == cls).float().view(-1) # creates binary mask where pixels belonging to the current class 'cls' are marked as 1 and others as 0
            targets = (targets == cls).float().view(-1) # create binary mask
            intersection = (predictions * targets).sum() # calculate the intersection for the current class
            union = predictions.sum() + targets.sum() # calculate the union for the current class
            dice_coeff = (2. * intersection + smooth)/(union + smooth) # calculate the dice_coeff for the current class
            dice_scores.append(dice_coeff.item()) # convert tensor to scalar and append to a list

        return tuple(dice_scores) # return the dice co-efficient as tuple
    """
    [1] LLRD: Layer-wise Learning Rate Decay
    [2] Weight Decay: L2-Regularization
    [3] Drop Path Rate (Stochastic Path): Randomly drops entire layers during training to help prevent
            overfitting and improve the robustness of the model.
    """
    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=0.0001,
                                    betas=(0.9,0.999),
                                    weight_decay=0.1) # set adam optimizer with 0.0001 learning rate
        elif self.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(self.parameters(),
                                     lr=0.0001,
                                     betas=(0.9,0.999),
                                     weight_decay=0.1) # set adamw optimizer with 0.0001 learning rate
        elif self.optimizer.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(),
                                       lr=0.0001,
                                       weight_decay=0.1) # set RMSProp optimizer with 0.0001 learning rate
        elif self.optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(),
                                   lr=0.0001,
                                   weight_decay=0.1) # set SGD optimizer with 0.0001 learning rate