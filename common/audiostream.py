import os
import time
import subprocess
import logging
import random

def get_audio_stream(request): 
    if 'audio' in request.files:
        audio_file = request.files['audio']
        
        base_filename = 'recording-'+str(random.randrange(0, 10000))+'-'+str(time.time())
        
        webm_filename = os.path.join('uploads', base_filename+'.webm')  
        mp3_filename = os.path.join('uploads', base_filename+'.mp3')  

        audio_file.save(webm_filename)

        # Convert webm to mp3
        subprocess.run([
            'ffmpeg',
            '-i', webm_filename,  # Input webm file
            '-vn',  # Disable video stream
            '-ar', '44100',  # Audio sampling rate (44.1 kHz)
            '-ac', '2',  # Number of audio channels (2 for stereo)
            '-b:a', '192k',  # Audio bitrate (192 kbps)
            mp3_filename  # Output mp3 file
        ])

        os.remove(webm_filename)  # Remove the temporary webm file
        logging.info('Audio saved successfully!')

        with open(mp3_filename, mode='rb') as file: # b is important -> binary
            data = file.read()
        file.close()

        # os.remove(mp3_filename)  # Remove the temporary mp3 file
        return data
    else:
        logging.info('No audio input was provided. Fallback to text.')
        return None
