# Stable Diffusion WebUI Forge/reForge

Stable Diffusion WebUI Forge/reForgeは、[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)（[Gradio](https://www.gradio.app/)ベース）上に構築されたプラットフォームで、開発を容易にし、リソース管理を最適化し、推論を高速化し、実験的な機能を研究するために作られました。

「Forge」という名前は「Minecraft Forge」から着想を得ています。このプロジェクトはSD WebUIのForgeになることを目指しています。

# Forge2/reForge2

詳細は https://github.com/Panchovix/stable-diffusion-webui-reForge/discussions/377#discussioncomment-14010687 をご覧ください。これらのブランチをここに残すか、「reForge2」のようにするかご意見をお聞かせください。

* newmain_newforge: 最新のforge2（gradio4、flux等）をベースに、ゆっくりと追加する予定の小さな変更を含みます。現在はPython 3.12サポート、sage/flashアテンションサポート、reForge(1)からの全サンプラーとスケジューラ、そして最近ではCFG++サンプラーのサポートがあります。
* newforge_dendev: @DenOfEquity (https://github.com/DenOfEquity/ersatzForge) による最新のersatzForgeフォーク（forge2ベース、gradio4、flux、chroma、cosmos、longclip、その他多数）をベースにしています。Denさん、reForgeであなたのフォークをベースに作業させていただきありがとうございます。古いreforgeからの新機能（全サンプラーなど）も追加していく予定です。

# 提案: 旧forgeベースの安定性を求めるなら、forge classicを使用してください

reForge(1)は残念ながら全てのタスクで安定しているわけではありません。

そのため、sd1.x、2.x、SDXLで旧forgeバックエンドをそのまま使い続けたい場合は、@Haoming02によるforge classicを使用することをお勧めします：https://github.com/Haoming02/sd-webui-forge-classic。現時点では、これが旧forgeの真の後継者です。

その他のブランチ:
* main: 複数の変更と更新を含むメインブランチ。ただしmain-oldブランチほど安定していません。
* dev: mainと似ていますが、より不安定な変更を含みます。例：A1111の代わりにcomfy/ldm_patchedバックエンドをsd1.xとsdxlに使用。
* dev2: devより不安定。現在はdevと同じ。
* experimental: dev2と同じですがgradio 4を使用。
* main-old: 旧forgeバックエンドを持つブランチ。おそらく最も安定した古いもの（2025-03）

# Forge/reForgeのインストール

### （推奨）クリーンインストール

Python（Python 3.7から3.12まで動作します。3.13はまだいくつかの問題があります）が必要です。
何をしているか分かっている場合は、SD-WebUIと同じ方法でForge/reForgeをインストールできます。（Git、Pythonをインストールし、reForgeリポジトリ `https://github.com/Panchovix/stable-diffusion-webui-reForge.git` をGit Cloneして、webui-user.batを実行）:

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
```
その後、webui-user.bat（Windows）またはwebui-user.sh（Linux。フォルダ、パス、必要な設定に応じて行のコメントを外してください）を実行します。

更新したい場合:
```bash
cd stable-diffusion-webui-reForge
git pull
```

### Windows 7やCUDA 11.xを使用している場合

別のrequirementsファイルを使用するため、インストール方法が少し異なります。元のreqファイルをバックアップにリネームし、レガシーのものを元の名前でコピーして、更新が動作するようにします。
Windows CMDの場合:

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
ren requirements_versions.txt requirements_versions_backup.txt
copy requirements_versions_legacy.txt requirements_versions.txt
```

Windows PS1の場合

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
Rename-Item requirements_versions.txt requirements_versions_backup.txt
Copy-Item requirements_versions_legacy.txt requirements_versions.txt
```

その後、webui-user.bat（Windows）を実行します。

### A1111を持っていてGitを知っている場合
チュートリアル元: https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git
すでにオリジナルのA1111を持っていてgitに精通している場合、`/path/to/stable-diffusion-webui` に移動して以下を実行するオプションがあります:
```bash
git remote add reForge https://github.com/Panchovix/stable-diffusion-webui-reForge
git branch Panchovix/main
git checkout Panchovix/main
git fetch reForge
git branch -u reForge/main
git stash
git pull
```
オリジナルのA1111に戻るには、`git checkout master` または `git checkout main` を実行するだけです。

マージで競合を解決する必要がある状態で止まった場合は、`git merge --abort` で戻れます。

-------

事前パッケージは計画されていますが、方法が分かりません。PRやヘルプを歓迎します。

# Forge/reForgeバックエンド

Forge/reForgeバックエンドは、リソース管理に関連するWebUIのすべてのコードを削除し、すべてを作り直しました。以前のCMDフラグ（`medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet`など）はすべて**削除**されました。これらのフラグを追加してもエラーにはなりませんが、何も行いません。

CMDフラグなしで、Forge/reForgeはSDXLを4GB VRAMで、SD1.5を2GB VRAMで実行できます。

**注意すべきフラグ:**

1. `--always-offload-from-vram`（このフラグは**遅く**なりますがリスクが低くなります）。このオプションはForge/reForgeに常にVRAMからモデルをアンロードさせます。複数のソフトウェアを一緒に使用してForge/reForgeのVRAM使用量を減らし他のソフトウェアにVRAMを与えたい場合、Forge/reForgeとVRAMを競合する古い拡張機能を使用している場合、または（非常にまれに）OOMが発生した場合に便利です。

2. `--cuda-malloc`（このフラグは**速く**なりますがリスクが高くなります）。これはpytorchにテンソルmallocに*cudaMallocAsync*を使用するよう要求します。一部のプロファイラーでミリ秒レベルのパフォーマンス向上を観察できますが、ほとんどのデバイスでの実際の速度向上は気づかないことが多いです（画像あたり約0.1秒以下）。多くのユーザーが非同期mallocがプログラムをクラッシュさせると報告しているため、デフォルトに設定できません。ユーザーは自己責任でこのcmdフラグを有効にする必要があります。

3. `--cuda-stream`（このフラグは**速く**なりますがリスクが高くなります）。これはpytorch CUDAストリーム（GPU上の特殊なスレッド）を使用してモデルの移動とテンソルの計算を同時に行います。これはほぼすべてのモデル移動時間を排除し、小さなVRAMを持つ30XX/40XXデバイス（例：RTX 4050 6GB、RTX 3060 Laptop 6GBなど）でSDXLを約15%から25%高速化できます。しかし、2060で純粋な黒い画像（Nan出力）の可能性が高くなること、1080と2060でOOMの可能性が高くなることを観察しているため、残念ながらデフォルトに設定できません。解像度が大きい場合、単一のアテンション層の計算時間がモデル全体をGPUに移動する時間より長くなる可能性があります。その場合、GPUがモデル全体で満たされているため次のアテンション層はOOMし、別のアテンション層を計算するための残りのスペースがありません。ほとんどのオーバーヘッド検出方法は古いデバイスで信頼性が十分ではありません（私のテストでは）。ユーザーは自己責任でこのcmdフラグを有効にする必要があります。

4. `--pin-shared-memory`（このフラグは**速く**なりますがリスクが高くなります）。`--cuda-stream`と一緒に使用した場合のみ有効です。モデルをオフロードする際、システムRAMの代わりに共有GPUメモリにモジュールをオフロードします。小さなVRAMを持つ一部の30XX/40XXデバイス（例：RTX 4050 6GB、RTX 3060 Laptop 6GBなど）で、SDXLの大幅な（少なくとも20%）速度向上を観察できます。しかし、共有GPUメモリのOOMは通常のGPUメモリOOMよりもはるかに深刻な問題であるため、残念ながらデフォルトに設定できません。Pytorchは共有GPUメモリをアンロードまたは検出する堅牢な方法を提供していません。共有GPUメモリがOOMになると、プログラム全体がクラッシュし（GTX 1060/1050/1066でSDXLで観察）、クラッシュを防止または回復する動的な方法はありません。ユーザーは自己責任でこのcmdフラグを有効にする必要があります。

パフォーマンスを向上させたり、VRAMを節約したりするのに役立つ追加フラグ。ほとんどはldm_patched/modules/args_parser.pyと通常のA1111パス（modules/cmd_args.py）にあります:

    --disable-xformers
        xformersを無効にし、SDPなどの他のアテンションを使用します。
    --use-sage-attention
        https://github.com/thu-ml/SageAttention からのSAGEアテンション実装を使用します。tritonが必要なため、ライブラリを別途インストールする必要があります。
    --attention-split
        分割クロスアテンション最適化を使用します。xformers使用時は無視されます。
    --attention-quad
        準二次クロスアテンション最適化を使用します。xformers使用時は無視されます。
    --attention-pytorch
        新しいpytorch 2.0クロスアテンション関数を使用します。
    --disable-attention-upcast
        すべてのアテンションのアップキャストを無効にします。デバッグ以外では不要です。
    --force-channels-last
        モデル推論時にチャンネルラストフォーマットを強制します。
    --disable-cuda-malloc
        cudaMallocAsyncを無効にします。
    --gpu-device-id
        このインスタンスが使用するcudaデバイスのIDを設定します。
    --force-upcast-attention
        アテンションのアップキャストを強制的に有効にします。

（VRAM関連）

    --always-gpu
        すべて（テキストエンコーダ/CLIPモデルなど）をGPUに保存して実行します。
    --always-high-vram
        デフォルトではモデルは使用後にCPUメモリにアンロードされます。このオプションはそれらをGPUメモリに保持します。
    --always-normal-vram
        lowvramが自動的に有効になった場合に通常のvram使用を強制するために使用します。
    --always-low-vram
        unetを分割してvramの使用量を減らします。
    --always-no-vram
        lowvramでも不十分な場合。
    --always-cpu
        すべてにCPUを使用します（遅い）。

（浮動小数点タイプ）

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

（レアなプラットフォーム）

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

# Lora ctl（コントロール）

このリポジトリをreforge用に適応させて追加しました。

これはオリジナルなしでは不可能でした！

Lora ctl（コントロール）のchealdに大きなクレジット。reforge拡張機能へのリンク: https://github.com/Panchovix/sd_webui_loractl_reforge_y.git

loraコントロールの予備的な動作バージョンの作業について@1rreに感謝します！

使用方法はそれぞれのリポジトリでご覧いただけます

https://github.com/cheald/sd-webui-loractl

## 組み込み拡張機能を別リポジトリに移動

UIが組み込み拡張機能で非常に煩雑になったため、一部を削除して別のリポジトリにしました。UIの拡張機能インストーラーでインストールするか、extensionsフォルダで`git clone repo.git`（`repo.git`を以下のリンクに置き換え）を実行してインストールできます。

* RAUNet-MSW-MSA (HiDiffusion): https://github.com/Panchovix/reforge_jankhidiffusion.git
* Skimmed CFG: https://github.com/Panchovix/reForge-SkimmedCFG.git
* Forge Style Align: https://github.com/Panchovix/sd_forge_stylealign.git
* reForge Sigmas Merge: https://github.com/Panchovix/reForge-Sigmas_merge.git
* Differential Diffusion: https://github.com/Panchovix/reForge-DifferentialDiffusion.git
* Automatic CFG: https://github.com/Panchovix/reForge-AutomaticCFG.git
* reForge_Advanced_CLIP_Text_Encode（まだ動作しません）: https://github.com/Panchovix/reForge_Advanced_CLIP_Text_Encode.git
* Hunyuan-DiT-for-webUI-main: https://github.com/Panchovix/Hunyuan-DiT-for-webUI-main.git
* PixArt-Sigma-for-webUI-main: https://github.com/Panchovix/PixArt-Sigma-for-webUI-main.git
* StableCascade-for-webUI-main: https://github.com/Panchovix/StableCascade-for-webUI-main.git
* StableDiffusion3-for-webUI-main: https://github.com/Panchovix/StableDiffusion3-for-webUI-main.git

# forge2以前の最後の「旧」Forgeコミット (https://github.com/lllyasviel/stable-diffusion-webui-forge/commit/bfee03d8d9415a925616f40ede030fe7a51cbcfd)

# サポート

プロジェクトへの寄付やサポート方法について質問をいただき、本当に感謝しています！いくつかの提案からbuymeacoffeのリンクを作成しました！

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Panchovix)
