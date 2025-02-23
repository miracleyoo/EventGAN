import os

def get_latest_checkpoint(save_dir):
    checkpoint_list = [] 
    print(save_dir)
    for dirpath, dirnames, filenames in os.walk(save_dir):
        
        for filename in filenames:
            if filename.endswith('.pt'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) == 0) else checkpoint_list[-1]
    print(latest_checkpoint)
    return latest_checkpoint
