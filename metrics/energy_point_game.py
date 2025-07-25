'''
Implementation of Energy-based Pointing Game proposed in Score-CAM.
'''

import torch

'''
bbox (list): upper left and lower right coordinates of object bounding box
saliency_map (array): explanation map, ignore the channel
'''
def EnergyPointGame(bbox, saliency_map):
  
  x1, y1, x2, y2 = bbox
  w, h = saliency_map.shape
  
  empty = torch.zeros((w, h))
  # empty[x1:x2, y1:y2] = 1
  empty[y1:y2, x1:x2] = 1
  mask_bbox = saliency_map * empty  
  
  energy_bbox =  mask_bbox.sum()
  energy_whole = saliency_map.sum()
  
  proportion = energy_bbox / energy_whole
  
  return proportion


def EnergyPointGame_Threshold(bbox, saliency_map, threshold=0.5):
    x1, y1, x2, y2 = bbox

    # Apply threshold
    saliency_thresh = saliency_map.clone()
    saliency_thresh[saliency_thresh < threshold] = 0.0

    # Create bounding box mask
    bbox_mask = torch.zeros_like(saliency_map)
    bbox_mask[x1:x2, y1:y2] = 1.0

    # Multiply and compute energy
    energy_bbox = (saliency_thresh * bbox_mask).sum()
    energy_total = saliency_thresh.sum()
    
    if energy_total.item() == 0:
      return 0 
    else: 
      return energy_bbox.item() / energy_total.item()