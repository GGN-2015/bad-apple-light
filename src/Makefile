all: final.mp4

clean:
	rm final.mp4 output.mp4

final.mp4: output.mp4 bin/output_audio.mp3
	ffmpeg -hwaccel cuda -i output.mp4 -i bin/output_audio.mp3 -c:v copy -c:a aac -strict experimental final.mp4

output.mp4: gen.py out_img.py var_img.py
	python gen.py
	ffmpeg -framerate 30 -i new_frames/output_frame_%04d.png -c:v libx265 -b:v 15M -minrate 12M -maxrate 18M -rc-lookahead 32 -cq 22 -preset medium -pix_fmt yuv420p output.mp4

.PHONY: all clean