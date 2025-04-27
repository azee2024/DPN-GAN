# import os
# import shutil
# import re
#
#
# def move_selected_files(source_dir, target_dir):
#     # Ensure the target directory exists
#     os.makedirs(target_dir, exist_ok=True)
#
#     # Compile the regex pattern to match files named {}_{}_6.npz to {}_{}_49.npz
#     pattern = re.compile(r'^[^_]+_[^_]+_([6-9]|[1-4][0-9])\.npz$')
#
#     # Iterate over files in the source directory
#     for filename in os.listdir(source_dir):
#         # Check if the filename matches the pattern
#         if pattern.match(filename):
#             # Construct full file paths
#             source_file = os.path.join(source_dir, filename)
#             target_file = os.path.join(target_dir, filename)
#             # Move the file
#             shutil.move(source_file, target_file)
#             # print(f"Moved: {source_file} to {target_file}")
#
#
# # Example usage
# source_dir = "data/audiomnist/preprocessed_data"
# target_dir = "data/audiomnist/substantial_set"
# move_selected_files(source_dir, target_dir)


lambda_mel = 0.7
lambda_feat = 0.2

gen_adv_loss = 0.25
mel_loss = 0.69
feat_loss = 0.0

msd_loss = 0.37
mcd_loss = 0.32

gen_loss = gen_adv_loss + lambda_mel * mel_loss + lambda_feat * feat_loss
disc_loss = msd_loss + mcd_loss


print("Generator Loss: ", gen_loss)

print("Adversarial Loss: ", gen_adv_loss)
print("Mel Spectrogram Loss: ", mel_loss)
print("Feature Matching Loss: ", feat_loss)

print("Discriminator Loss: ", disc_loss)
print("MSD Loss: ", msd_loss)
print("MCD Loss: ", mcd_loss)
