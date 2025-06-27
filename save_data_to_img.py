import zarr
import cv2
import os

data_name = '0'  # 0, 1, 2
data_zarr = zarr.open(f'./data/{data_name}/data_buffer.zarr', mode='r')
print(data_zarr.tree())

img = data_zarr['cam_0']['data']['color'][:][10]
depth = data_zarr['cam_0']['data']['depth'][:][10]
timestamp = data_zarr['cam_0']['data']['timestamp'][:][10]
print(img.shape)
print(depth.shape)
print(timestamp.shape)

print(data_zarr['cam_0']['meta']['intrinsics'][:])

imgs = data_zarr['cam_0']['data']['color'][:]
depths = data_zarr['cam_0']['data']['depth'][:]
timestamps = data_zarr['cam_0']['data']['timestamp'][:]

for i in range(len(imgs)):
    print(i)
    img = imgs[i]
    depth = depths[i]
    timestamp = timestamps[i]
    timestamp = int(timestamp * 1e6)
    print(timestamp)
    os.makedirs(f'./results/{data_name}/rgb', exist_ok=True)
    os.makedirs(f'./results/{data_name}/depth', exist_ok=True)
    img_path = f'./results/{data_name}/rgb/{timestamp}.png'
    depth_path = f'./results/{data_name}/depth/{timestamp}.png'
    cv2.imwrite(img_path, img)
    cv2.imwrite(depth_path, depth)