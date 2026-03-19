#!/bin/bash 
export HF_ENDPOINT=https://hf-mirror.com 
export HUGGINGFACE_TOKEN=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

#mkdir -p /home/fit/renju/WORK/jty/vita_data
source /home/fit/renju/WORK/miniconda3/bin/activate
conda init
conda activate jty_1
huggingface-cli login
huggingface-cli download --repo-type dataset --resume-download amphion/Emilia-Dataset --local-dir /home/fit/renju/WORK/jty/vita_data/amphion
huggingface-cli download --repo-type dataset --resume-download fixie-ai/llama-questions --local-dir /home/fit/renju/WORK/jty/vita_data/fixie/llama-questions
huggingface-cli download --repo-type dataset --resume-download fixie-ai/librispeech_asr --local-dir /home/fit/renju/WORK/jty/vita_data/fixie/librispeech_asr
huggingface-cli download --repo-type dataset --resume-download fixie-ai/trivia_qa-audio --local-dir /home/fit/renju/WORK/jty/vita_data/fixie/trivia_qa-audio
huggingface-cli download --repo-type dataset --resume-download facebook/voxpopuli --local-dir /home/fit/renju/WORK/jty/vita_data/facebook/voxpopuli
huggingface-cli download --repo-type dataset --resume-download MushanW/GLOBE_V2 --local-dir /home/fit/renju/WORK/jty/vita_data/MushanW/GLOBE_V2
huggingface-cli download --repo-type dataset --resume-download MLCommons/peoples_speech --local-dir /home/fit/renju/WORK/jty/vita_data/MLCommons/peoples_speech
huggingface-cli download --repo-type dataset --resume-download fsicoli/common_voice_17_0 --local-dir /home/fit/renju/WORK/jty/vita_data/fsicoli/common_voice_17_0
huggingface-cli download --repo-type dataset --resume-download mythicinfinity/libritts_r --local-dir /home/fit/renju/WORK/jty/vita_data/mythicinfinity/libritts_r
huggingface-cli download --repo-type dataset --resume-download mythicinfinity/libritts --local-dir /home/fit/renju/WORK/jty/vita_data/mythicinfinity/libritts
#wget -P /home/fit/renju/WORK/jty/vita_data/openslr https://openslr.trmal.net/resources/68/train_set.tar.gz
#wget -P /home/fit/renju/WORK/jty/vita_data/openslr https://openslr.trmal.net/resources/68/dev_set.tar.gz
#wget -P /home/fit/renju/WORK/jty/vita_data/openslr https://openslr.trmal.net/resources/68/test_set.tar.gz
huggingface-cli download --repo-type dataset --resume-download parler-tts/mls_eng --local-dir /home/fit/renju/WORK/jty/vita_data/parler-tts/mls_eng
huggingface-cli download --repo-type dataset --resume-download shenyunhang/AISHELL-1 --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/AISHELL-1
#huggingface-cli download --repo-type dataset --resume-download shenyunhang/AISHELL-2 --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/AISHELL-2
huggingface-cli download --repo-type dataset --resume-download shenyunhang/AISHELL-3 --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/AISHELL-3
huggingface-cli download --repo-type dataset --resume-download shenyunhang/AISHELL-4 --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/AISHELL-4
huggingface-cli download --repo-type dataset --resume-download shenyunhang/VoiceAssistant-400K --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/VoiceAssistant-400K
huggingface-cli download --repo-type dataset --resume-download shenyunhang/AudioQA-1M --local-dir /home/fit/renju/WORK/jty/vita_data/shenyunhang/AudioQA-1M
#wget -P /home/fit/renju/WORK/jty/vita_data/openslr https://openslr.trmal.net/resources/33/data_aishell.tgz
#wget -P /home/fit/renju/WORK/jty/vita_data/openslr https://openslr.trmal.net/resources/33/resource_aishell.tgz
huggingface-cli download --repo-type dataset --resume-download speechcolab/gigaspeech --local-dir /home/fit/renju/WORK/jty/vita_data/speechcolab/gigaspeech
huggingface-cli download --repo-type dataset --resume-download wenet-e2e/wenetspeech --local-dir /home/fit/renju/WORK/jty/vita_data/wenet-e2e/wenetspeech
huggingface-cli download --repo-type dataset --resume-download Wenetspeech4TTS/WenetSpeech4TTS --local-dir /home/fit/renju/WORK/jty/vita_data/Wenetspeech4TTS/WenetSpeech4TTS
