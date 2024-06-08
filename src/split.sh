#!/bin/bash

mkdir -p ./frames
mkdir -p ./new_frames
ffmpeg -i ./bin/badapple.mp4 -vf fps=14.985 frames/output_frame_%04d.png
