import argparse
import os
import glob
import cv2
import numpy as np

from pipeline.preprocessing import bgr_to_ycbcr
from pipeline.intra_coding import encode_iframe, decode_iframe
from pipeline.inter_coding import encode_pframe, decode_pframe
from pipeline.entropy_coding import write_bitstream

def main(frames_dir, output_path, GOP=8, Q=1.0, search_range=8):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")) +
                         glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    
    metadata = {
        "num_frames": len(frame_paths),
        "frame_height": height,
        "frame_width": width,
        "GOP": GOP,
        "Q": Q,
        "search_range": search_range,
    }
    
    frames_data = []
    
    print(f"Encoding {metadata['num_frames']} frames from {frames_dir}...")
    
    recon_Y = None

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"Warning: Could not read {path}, skipping.")
            continue
            
        is_iframe = (i % GOP == 0)
        
        
        
        if is_iframe:
            Y, Cb, Cr = bgr_to_ycbcr(frame)
            Y_coeffs = encode_iframe(Y, Q)
            Cb_coeffs = encode_iframe(Cb, Q)
            Cr_coeffs = encode_iframe(Cr, Q)
            
            frame_data = {
                "type": "I",
                "Y_coeffs": Y_coeffs,
                "Cb_coeffs": Cb_coeffs,
                "Cr_coeffs": Cr_coeffs
            }
            frames_data.append(frame_data)
            
            recon_Y = decode_iframe(Y_coeffs, Y.shape, Q)
            print(f"Encoded frame {i} as I-frame.")
        else:
            Y, Cb, Cr = bgr_to_ycbcr(frame)
            encoded_p = encode_pframe(Y, recon_Y, Q, search_range)
            Cb_coeffs = encode_iframe(Cb, Q)
            Cr_coeffs = encode_iframe(Cr, Q)
            
            frame_data = {
                "type": "P",
                "motion_vectors": encoded_p["motion_vectors"],
                "Y_residuals": encoded_p["residual_coeffs"],
                "Cb_coeffs": Cb_coeffs,
                "Cr_coeffs": Cr_coeffs
            }
            frames_data.append(frame_data)
            
            recon_Y = decode_pframe(encoded_p, recon_Y, Q)
            print(f"Encoded frame {i} as P-frame.")

    bitstream = {
        "metadata": metadata,
        "frames": frames_data
    }
    write_bitstream(bitstream, output_path)
    print(f"Encoded bitstream saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="frames/")
    parser.add_argument("--output", default="output/video.bin")
    parser.add_argument("--GOP", type=int, default=8)
    parser.add_argument("--Q", type=float, default=1.0)
    parser.add_argument("--search_range", type=int, default=8)
    args = parser.parse_args()
    main(args.frames, args.output, args.GOP, args.Q, args.search_range)
