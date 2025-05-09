import torch

def save_state(run_folder, prefix="boat_state", boat=None, global_step=None, epoch=None):
    """
    Save the full training state including boat (containing model weights, optimizer states),
    scheduler states, and training metadata.
    
    Args:
        run_folder: Path to the run folder
        boat: The boat being trained (contains models dictionary)
        global_step (int, optional): Current global step
        epoch (int, optional): Current epoch
    """

    run_folder.mkdir(parents=True, exist_ok=True)

    # Determine which identifier to use in the filename
    if global_step is not None and epoch is not None:
        state_path = run_folder / f"{prefix}_step={global_step}.pt"
        print(f"Warning: Both global_step and epoch provided. Using global_step ({global_step}) for filename.")
    elif global_step is not None:
        state_path = run_folder / f"{prefix}_step={global_step}.pt"
    elif epoch is not None:
        state_path = run_folder / f"{prefix}_epoch={epoch}.pt"
    else:
        state_path = run_folder / "latest_state.pt"
        print("Warning: Neither global_step nor epoch provided. Using generic filename.")
    
    # Save models individually
    networks_state = {}
    for name, model in boat.models.items():
        networks_state[name] = model.state_dict()
    
    # Prepare the state dictionary
    state = {
        'networks_state': networks_state,
    }
    
    # Add tracking variables to state if they exist
    if global_step is not None:
        state['global_step'] = global_step
    if epoch is not None:
        state['epoch'] = epoch
    
    # Add optimizer state (part of the boat)
    if hasattr(boat, 'optimizer'):
        state['optimizer_state'] = boat.optimizer.state_dict()
    
    # Add scheduler state if it exists in the boat
    if hasattr(boat, 'scheduler') and boat.scheduler is not None:
        state['scheduler_state'] = boat.scheduler.state_dict()
    
    # Save training configuration, placeholder for now
    state['trainer_config'] = {}
    
    # Save the state
    torch.save(state, state_path)
    print(f"Full training state saved to {state_path}")
    
    return state_path

def load_state(state_path, boat, strict=True):
    """
    Load the full training state including boat model weights, optimizer states,
    scheduler states, and training metadata.
    
    Args:
        state_path (Path): Path to the saved state
        boat: The boat to load weights into (contains models dictionary)
        
    Returns:
        dict: Training metadata (epoch, global_step, etc.)
    """

    if not state_path.exists():
        raise FileNotFoundError(f"No state file found at {state_path}")
    
    # Load the state dictionary
    state = torch.load(state_path)
    
    # Load model weights
    networks_state = state['networks_state']
    for name, state_dict in networks_state.items():
        if name in boat.models:
            boat.models[name].load_state_dict(state_dict, strict=strict)
        else:
            print(f"Warning: Network {name} in saved state not found in boat")
    
    # Load optimizer state
    if hasattr(boat, 'optimizer') and 'optimizer_state' in state:
        boat.optimizer.load_state_dict(state['optimizer_state'], strict=strict)
    
    # Load scheduler state if it exists
    if hasattr(boat, 'scheduler') and boat.scheduler is not None and 'scheduler_state' in state:
        boat.scheduler.load_state_dict(state['scheduler_state'], strict=strict)
    
    # Get training metadata
    metadata = {
        'global_step': state.get('global_step', 0),
        'epoch': state.get('epoch', 0),
    }
    
    print(f"Full training state loaded from {state_path}")
    print(f"Resuming from epoch {metadata['epoch']} and step {metadata['global_step']}")
    
    return boat, metadata