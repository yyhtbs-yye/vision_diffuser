import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def log_images(
    logger,
    images_dict,
    keys_to_log,
    global_step,
    normalize=True,
    prefix="val",
    max_images=4
):
    """Log images to the provided logger.
    
    Args:
        logger: The logger object that has an add_image method (like TensorBoard)
        images_dict: Dictionary with image tensors of shape [B, C, H, W]
        keys_to_log: List of keys from images_dict to visualize
        global_step: Current global step for logging
        normalize: Whether to normalize images from [-1, 1] to [0, 1]
        prefix: Prefix for the logged image names
        max_images: Maximum number of images to include from each batch
    """
    for key in keys_to_log:
        if key not in images_dict:
            print(f"Warning: Key '{key}' not found in images_dict. Skipping.")
            continue
        
        # Get image tensor
        images = images_dict[key]
        
        # Apply normalization if needed
        if normalize:
            images = (images.clamp(-1, 1) + 1) / 2
            
        # Limit number of images
        batch_size = min(max_images, images.shape[0])
        images = images[:batch_size]
        
        # Flatten the batch dimension along width: [B, C, H, W] -> [C, H, B*W]
        # First, rearrange to [C, B, H, W]
        images = images.permute(1, 0, 2, 3)
        
        # Get dimensions
        C, B, H, W = images.shape
        
        # Reshape to [C, H, B*W]
        images = images.reshape(C, H, B*W)
        
        # Log to the logger
        logger.add_image(
            f"{prefix}/{key}",
            images,
            global_step
        )


def visualize_comparisons(
    logger,
    images_dict,
    keys,
    global_step,
    wnb=(0, 1),
    prefix="val",
    max_images=4,
    texts=None,
):
    """Create and log comparison visualizations between different image types.
    
    Args:
        logger: The logger object with an add_image method
        images_dict: Dictionary with image tensors of shape [B, C, H, W]
        keys: List of lists, where each inner list contains keys to compare side by side
        global_step: Current global step for logging
        normalize: Whether to normalize images from [-1, 1] to [0, 1]
        prefix: Prefix for the logged image names
        max_images: Maximum number of images to include from each batch
    """
    for comp_idx, key in enumerate(keys):
        # Skip if any key is missing
        if key not in images_dict:
            print(f"Warning: Keys {key} not found in images_dict. Skipping comparison.")
            continue
        
        # Create a name for this comparison
        comparison_name = f"{prefix}/visualization-{key}"
        
        # Normalize images if needed
        images_to_compare = []
        img = images_dict[key]
        img = (img.clamp(-1, 1) * wnb[0]) + wnb[1]
        images_to_compare.append(img)
        
        # Ensure all images have the same batch size and dimensions
        batch_size = min(max_images, min(img.shape[0] for img in images_to_compare))
        images_to_compare = [img[:batch_size] for img in images_to_compare]
        
        # Create comparison grid - first horizontally concatenate pairs in each batch item
        comparison_rows = []
        for b in range(batch_size):
            # Concatenate images horizontally for this batch item
            row_images = [img[b] for img in images_to_compare]

            # Add text to each image if provided
            if texts is not None and b < len(texts):
                for i, img in enumerate(row_images):
                    # Convert tensor to PIL Image
                    img_np = img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
                    img_np = (img_np * 255).astype(np.uint8)  # Scale to 0-255
                    pil_img = Image.fromarray(img_np)
                    
                    # Draw text
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", size=8)
                    except:
                        font = ImageFont.load_default(size=6)  # Fallback to default font
                    # draw.rectangle((0, 0, 100, 15), fill=(0, 0, 0, 128))  # Semi-transparent background
                    draw.text((0, 0), texts[b], fill=(255, 0, 0), font=font)  # White text
                    
                    # Convert back to tensor
                    img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1) / 255.0
                    row_images[i] = img_tensor
            
            row = torch.cat(row_images, dim=2)  # Concatenate along width (dim=2)
            comparison_rows.append(row)
        
        # Then vertically concatenate all batch items
        comparison = torch.cat(comparison_rows, dim=1)  # Concatenate along height (dim=1)
        
        # Log to the logger
        logger.add_image(
            comparison_name,
            comparison,
            global_step
        )


def log_grid_images(
    logger,
    images,
    name,
    global_step,
    normalize=True,
    nrow=None
):
    """Log a grid of images to the provided logger.
    
    This function is useful for visualizing a large number of images in a grid format.
    
    Args:
        logger: The logger object with an add_image method
        images: Tensor of shape [B, C, H, W] with images to visualize
        name: Name for the logged images
        global_step: Current global step for logging
        normalize: Whether to normalize images from [-1, 1] to [0, 1]
        nrow: Number of images per row (defaults to square root of batch size)
    """
    # Apply normalization if needed
    if normalize:
        images = (images.clamp(-1, 1) + 1) / 2
    
    # Default to a square-ish grid if nrow not specified
    if nrow is None:
        from math import sqrt, ceil
        nrow = ceil(sqrt(images.shape[0]))
    
    # Use torchvision.utils.make_grid if available, otherwise roll our own
    try:
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=nrow)
    except ImportError:
        # Create our own grid
        b, c, h, w = images.shape
        rows = [images[i:i+nrow] for i in range(0, b, nrow)]
        rows = [torch.cat([img for img in row], dim=2) for row in rows]  # Cat along width
        grid = torch.cat(rows, dim=1)  # Cat along height
    
    # Log to the logger
    logger.add_image(name, grid, global_step)


def visualize_latent_space(
    logger,
    latents,
    name: str,
    global_step: int,
    channel_dim: int = 0,
    max_images: int = 16
):
    """Visualize latent space representations.
    
    Args:
        logger: The logger object with an add_image method
        latents: Tensor of latent representations [B, C, H, W]
        name: Name for the logged images
        global_step: Current global step for logging
        channel_dim: Which channel dimension to visualize (default is 0)
        max_images: Maximum number of images to include
    """
    # Limit batch size
    batch_size = min(max_images, latents.shape[0])
    latents = latents[:batch_size]
    
    # Extract the specified channel
    channel = latents[:, channel_dim:channel_dim+1]
    
    # Normalize for visualization
    channel_min = channel.min()
    channel_max = channel.max()
    if channel_min != channel_max:
        channel = (channel - channel_min) / (channel_max - channel_min)
    
    # Repeat to make RGB (grayscale)
    channel_rgb = channel.repeat(1, 3, 1, 1)
    
    # Log individual latent visualizations
    log_grid_images(
        logger=logger,
        images=channel_rgb,
        name=f"{name}_channel_{channel_dim}",
        global_step=global_step,
        normalize=False  # Already normalized
    )