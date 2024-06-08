all: final.mp4

clean:
	rm final.mp4 output.mp4

final.mp4: output.mp4 bin/output_audio.mp3
	ffmpeg -hwaccel cuda -i output.mp4 -i bin/output_audio.mp3 -c:v copy -c:a aac -strict experimental final.mp4

output.mp4: gen.py out_img.py var_img.py
	python gen.py
	ffmpeg -hwaccel cuda -r 14.985 -i new_frames/output_frame_%04d.png -vcodec libx265 -pix_fmt yuv444p -crf 0 output.mp4

.PHONY: all clean