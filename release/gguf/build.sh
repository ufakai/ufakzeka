# GGUF build of the ufakzeka-1 release checkpoint (internal id v103s5), run on the Hetzner box with the patched llama.cpp.
set -x; export PATH=$HOME/.local/bin:$PATH; cd /opt/llama.cpp; G=/data/ufakzeka/gguf; N=ufakzeka-1-instruct-v103s5
rm -rf $G/v103s5hf; mkdir -p $G/v103s5hf/hf
for f in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json; do modal volume get ufakzeka-ckpt $N/hf/$f $G/v103s5hf/hf/$f --force; done
/opt/ggufenv/bin/python convert_hf_to_gguf.py $G/v103s5hf/hf --outfile $G/$N-f16.gguf --outtype f16 2>&1 | grep -E "pre|successfully|Error"
./build/bin/llama-quantize $G/$N-f16.gguf $G/$N-q8_0.gguf q8_0 2>&1 | tail -1
ls -la $G/$N-*.gguf; echo V103S5_GGUF_DONE
