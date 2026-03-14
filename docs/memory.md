# Memory — WAN 2.2 Video PoC Notebook

## Objetivo

Criar um notebook Python para gerar e estender vídeos a partir de uma imagem usando o modelo WAN 2.2 Image-to-Video, encadeando segmentos pelo último frame de cada vídeo anterior.

---

## Arquivo gerado

`notebooks/wan_video_poc.ipynb`

---

## Decisões de arquitetura

### Modelo escolhido: WAN 2.2 I2V fp8 (Comfy-Org Repackaged)

Foram escolhidos os modelos fp8 do repositório `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` por serem ~50% menores que os fp16 originais:

| Componente             | Arquivo                                                             | Tamanho  |
| ---------------------- | ------------------------------------------------------------------- | -------- |
| Transformer High-Noise | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`                  | ~14.3 GB |
| Transformer Low-Noise  | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`                   | ~14.3 GB |
| VAE                    | `wan_2.1_vae.safetensors`                                           | ~1.5 GB  |
| Text Encoder           | `umt5_xxl_fp8_e4m3fn_scaled.safetensors`                            | ~5 GB    |
| Image Encoder (CLIP)   | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` (subfolder: `image_encoder`) | ~1 GB    |

O WAN 2.1 bf16 original pesava ~28–30 GB num único arquivo. Os dois fp8 juntos somam ~28.6 GB, mas cada arquivo individual é menor e o consumo de VRAM em execução é ~metade.

### Arquitetura de dois estágios (WAN 2.2)

O WAN 2.2 separa o denoising em dois transformers especializados:

```
Ruído puro                                     Vídeo final
[t=1000 ──────── t=500] [t=500 ─────────── t=0]
     ↑ transformer_high        ↑ transformer_low
  (estrutura/movimento)     (detalhes/qualidade)
```

- `SPLIT_TIMESTEP = 500` controla o ponto de troca
- `transformer_high`: gera estrutura e movimento (timesteps altos)
- `transformer_low`: refina detalhes visuais (timesteps baixos)

---

## Problema resolvido: estouro de memória RAM

### Causa

Carregar os dois transformers em `bfloat16` simultaneamente na RAM causava pico de ~56 GB (2 × 28 GB).

### Solução: `WAN22LazyDualTransformer`

Classe `nn.Module` que implementa lazy loading com swap sob demanda:

- Apenas **UM** transformer ocupa memória por vez
- Ao trocar, o transformer ativo é deletado → `gc.collect()` → `torch.cuda.empty_cache()` → antes de carregar o próximo
- A troca ocorre **uma única vez** por geração (quando o timestep cruza `SPLIT_TIMESTEP`)
- O `_TransformerConfigProxy` expõe a configuração JSON como atributos para compatibilidade com o pipeline

**Perfil de memória com lazy loading:**

| Fase                   | RAM                               | VRAM   |
| ---------------------- | --------------------------------- | ------ |
| Construção do pipeline | ~0 GB                             | ~0 GB  |
| `preload("high")`      | pico ~28 GB → liberado            | ~28 GB |
| Passos 1–20 (t > 500)  | ~0 GB                             | ~28 GB |
| Troca para low-noise   | del high → pico ~28 GB → liberado | ~28 GB |
| Passos 21–40           | ~0 GB                             | ~28 GB |

---

## Otimização de downloads: hf-transfer

- Adicionado `hf-transfer>=0.1.8` na célula de instalação
- `HF_HUB_ENABLE_HF_TRANSFER=1` definido antes de qualquer import do `huggingface_hub`
- Acelera downloads em até **10×** via transferências paralelas em Rust
- `hf_hub_download` com `local_dir_use_symlinks=False` evita redownloads se o arquivo já existe

---

## Estrutura do notebook (células)

| Célula | Seção                   | Conteúdo                                                                           |
| ------ | ----------------------- | ---------------------------------------------------------------------------------- |
| 1      | Título                  | Descrição do PoC, tabela de modelos, referências                                   |
| 2      | 1. Instalação           | `pip install` com `hf-transfer`; define `HF_HUB_ENABLE_HF_TRANSFER=1`              |
| 3      | 2. Imports              | `torch`, `diffusers`, `imageio`, `PIL`, etc.                                       |
| 4      | 3. Configuração         | `HF_REPO_ID`, `HIGH/LOW_NOISE_MODEL_FILE`, `SPLIT_TIMESTEP`, parâmetros de geração |
| 5      | 3.5 Download (markdown) | Tabela de arquivos e tamanhos                                                      |
| 6      | 3.5 Download (código)   | `download_wan22_model_files()` com `hf_hub_download`                               |
| 7      | 4. Load (markdown)      | Documenta arquitetura de dois estágios                                             |
| 8      | 4. Load (código)        | `_TransformerConfigProxy`, `WAN22LazyDualTransformer`, `load_wan22_pipeline()`     |
| 9      | 5. Pré-processamento    | `preprocess_image()` — resize com `mod_value` e aspect ratio                       |
| 10     | 6. Geração inicial      | `generate_video_from_image()` → `segment_000.mp4`                                  |
| 11     | 7. Extração de frames   | `extract_last_frames()`, `get_last_frame_as_image()`                               |
| 12     | 8. Extensão             | `extend_video_segment()` — usa último frame como conditioning image                |
| 13     | 9. Loop de extensão     | Itera `EXTENSION_LOOPS` vezes, incrementando seed                                  |
| 14     | 10. Concatenação        | `concatenate_video_segments()` (imageio) + `concatenate_segments_with_ffmpeg()`    |
| 15     | 11. Exibição            | `display_video()`, `print_generation_summary()`                                    |
| 16     | Bônus                   | Exibe cada segmento individualmente                                                |

---

## Técnica de extensão de vídeo

1. Gera segmento inicial a partir da imagem → `segment_000.mp4`
2. Extrai o último frame do segmento anterior como `PIL.Image`
3. Usa esse frame como `image` conditioning no próximo `WanImageToVideoPipeline()`
4. Descarta os primeiros `OVERLAP_FRAMES` do novo segmento na concatenação final (evita duplicatas)
5. Incrementa `seed = SEED + loop_index` para variação controlada entre segmentos

**Parâmetros de sobreposição:**

- 8–12 frames: transições rápidas, menor consistência
- 16–20 frames: equilíbrio recomendado (`OVERLAP_FRAMES=16` padrão)
- 24–32 frames: alta consistência, segmentos efetivos mais curtos

---

## Referências consultadas

- [ComfyUI WAN 2.2 Examples](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)
- [Comfy-Org WAN 2.2 Repackaged (HuggingFace)](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)
- [WAN 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [WAN Video Extender (ComfyUI)](https://github.com/Granddyser/wan-video-extender)
- [Diffusers WanImageToVideoPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan)
