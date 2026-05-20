import argparse
import os
import cv2

from pipeline.entropy_coding import read_bitstream
from pipeline.intra_coding import decode_iframe
from pipeline.inter_coding import decode_pframe
from pipeline.preprocessing import ycbcr_to_bgr

def main(bin_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bitstream = read_bitstream(bin_path)
    
    metadata = bitstream["metadata"]
    frames = bitstream["frames"]
    
    Q = metadata["Q"]
    orig_shape_y = (metadata["frame_height"], metadata["frame_width"])
    orig_shape_c = (metadata["frame_height"] // 2, metadata["frame_width"] // 2)
    
    print(f"Decoding {len(frames)} frames...")
    
    ref_Y = None
    
    for i, frame_data in enumerate(frames):
        if frame_data["type"] == "I":
            Y_recon = decode_iframe(frame_data["Y_coeffs"], orig_shape_y, Q)
            Cb_recon = decode_iframe(frame_data["Cb_coeffs"], orig_shape_c, Q)
            Cr_recon = decode_iframe(frame_data["Cr_coeffs"], orig_shape_c, Q)
            
            ref_Y = Y_recon
            
            bgr = ycbcr_to_bgr(Y_recon, Cb_recon, Cr_recon)
            out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(out_path, bgr)
            print(f"Decoded I-frame to {out_path}")
            
        elif frame_data["type"] == "P":
            encoded_p = {
                "motion_vectors": frame_data["motion_vectors"],
                "residual_coeffs": frame_data["Y_residuals"]
            }
            Y_recon = decode_pframe(encoded_p, ref_Y, Q)
            Cb_recon = decode_iframe(frame_data["Cb_coeffs"], orig_shape_c, Q)
            Cr_recon = decode_iframe(frame_data["Cr_coeffs"], orig_shape_c, Q)
            
            ref_Y = Y_recon
            
            bgr = ycbcr_to_bgr(Y_recon, Cb_recon, Cr_recon)
            out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(out_path, bgr)
            print(f"Decoded P-frame to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/video.bin")
    parser.add_argument("--output_dir", default="output/frames/")
    args = parser.parse_args()
    main(args.input, args.output_dir)
