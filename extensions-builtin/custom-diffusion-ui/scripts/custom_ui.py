"""
Custom Diffusion UI - プリセット管理機能付きカスタムインターフェース
"""

import os
import json
import base64
import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple

import gradio as gr
from PIL import Image

from modules import script_callbacks, shared, sd_samplers, sd_schedulers
from modules.processing import StableDiffusionProcessingTxt2Img, process_images, Processed
from modules.ui_components import FormRow


# プリセットディレクトリのパス
PRESET_DIR = Path(__file__).parent.parent / "presets"
PRESET_FILE = PRESET_DIR / "history.json"
THUMBNAIL_DIR = PRESET_DIR / "thumbnails"

# ディレクトリを作成
PRESET_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)


class PresetManager:
    """プリセット管理クラス"""

    def __init__(self):
        self.presets: List[Dict[str, Any]] = []
        self.load_presets()

    def load_presets(self):
        """プリセットをファイルから読み込み"""
        if PRESET_FILE.exists():
            try:
                with open(PRESET_FILE, 'r', encoding='utf-8') as f:
                    self.presets = json.load(f)
            except Exception as e:
                print(f"プリセット読み込みエラー: {e}")
                self.presets = []
        else:
            self.presets = []

    def save_presets(self):
        """プリセットをファイルに保存"""
        try:
            with open(PRESET_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"プリセット保存エラー: {e}")

    def add_preset(self, name: str, settings: Dict[str, Any], image: Image.Image) -> str:
        """新しいプリセットを追加"""
        timestamp = datetime.datetime.now().isoformat()
        preset_id = f"preset_{len(self.presets)}_{int(datetime.datetime.now().timestamp())}"

        # サムネイル画像を保存
        thumbnail_path = THUMBNAIL_DIR / f"{preset_id}.png"
        thumbnail = image.copy()
        thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
        thumbnail.save(thumbnail_path)

        preset = {
            "id": preset_id,
            "name": name,
            "timestamp": timestamp,
            "thumbnail": str(thumbnail_path),
            "settings": settings
        }

        self.presets.insert(0, preset)  # 最新のものを先頭に
        self.save_presets()

        return f"プリセット '{name}' を保存しました！"

    def get_preset(self, preset_id: str) -> Dict[str, Any]:
        """プリセットIDから設定を取得"""
        for preset in self.presets:
            if preset["id"] == preset_id:
                return preset
        return None

    def get_gallery_data(self) -> List[Tuple[str, str]]:
        """ギャラリー表示用のデータを取得"""
        gallery_data = []
        for preset in self.presets:
            if Path(preset["thumbnail"]).exists():
                gallery_data.append((
                    preset["thumbnail"],
                    f"{preset['name']}\n{preset['timestamp'][:10]}"
                ))
        return gallery_data


# グローバルプリセットマネージャー
preset_manager = PresetManager()


def get_available_samplers() -> List[str]:
    """利用可能なサンプラーのリストを取得"""
    return ["DPM++ 2M", "DPM++ 3M SDE", "Euler a"]


def get_available_schedulers() -> List[str]:
    """利用可能なスケジューラーのリストを取得"""
    return ["Karras", "SGM Uniform"]


def get_available_upscalers() -> List[str]:
    """利用可能なアップスケーラーのリストを取得"""
    upscalers = ["None"]

    # 全てのアップスケーラーから指定されたものを探す
    for upscaler in shared.sd_upscalers:
        name = upscaler.name
        # 要求されたアップスケーラーを探す
        if "4x-UltraSharp" in name or "UltraSharp" in name:
            upscalers.append(name)
        elif "DAT" in name and "x4" in name:
            upscalers.append(name)
        elif "ESRGAN" in name and "4x" in name and "Anime6B" in name:
            upscalers.append(name)

    # デフォルトのアップスケーラーも追加
    if "Latent" not in upscalers:
        upscalers.append("Latent")
    if "DAT x4" not in upscalers:
        upscalers.append("DAT x4")

    return upscalers


def generate_image(
    prompt: str,
    negative_prompt: str,
    sampler_name: str,
    scheduler: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    batch_count: int,
    batch_size: int,
    enable_hr: bool,
    hr_scale: float,
    hr_upscaler: str,
    hr_steps: int,
    denoising_strength: float,
    # ADetailer風の設定
    enable_face_restore: bool,
    face_restore_strength: float,
    adetailer_conf_1: float,
    adetailer_conf_2: float,
    adetailer_conf_3: float,
    adetailer_conf_4: float,
) -> Tuple[List[Image.Image], str, str]:
    """画像を生成"""

    # 固定モデル設定の確認（オーバーライド）
    override_settings = {}

    # 画像生成パラメータを設定
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_name=sampler_name,
        scheduler=scheduler,
        batch_size=batch_size,
        n_iter=batch_count,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_steps,
        denoising_strength=denoising_strength,
        override_settings=override_settings,
    )

    # フェイスリストア設定
    if enable_face_restore:
        p.restore_faces = True
        # ADetailer風の設定をメタデータに追加
        p.extra_generation_params["ADetailer Conf 1st"] = adetailer_conf_1
        p.extra_generation_params["ADetailer Conf 2nd"] = adetailer_conf_2
        p.extra_generation_params["ADetailer Conf 3rd"] = adetailer_conf_3
        p.extra_generation_params["ADetailer Conf 4th"] = adetailer_conf_4
        p.extra_generation_params["Face Restore Strength"] = face_restore_strength

    # 画像生成実行
    processed: Processed = process_images(p)

    # 結果を返す
    images = processed.images
    info = processed.info
    html_info = processed.comments_html

    return images, info, html_info


def save_preset(
    preset_name: str,
    gallery_images: List,
    prompt: str,
    negative_prompt: str,
    sampler_name: str,
    scheduler: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    batch_count: int,
    batch_size: int,
    enable_hr: bool,
    hr_scale: float,
    hr_upscaler: str,
    hr_steps: int,
    denoising_strength: float,
    enable_face_restore: bool,
    face_restore_strength: float,
    adetailer_conf_1: float,
    adetailer_conf_2: float,
    adetailer_conf_3: float,
    adetailer_conf_4: float,
) -> Tuple[str, List]:
    """現在の設定をプリセットとして保存"""

    if not preset_name:
        return "エラー: プリセット名を入力してください", preset_manager.get_gallery_data()

    if not gallery_images or len(gallery_images) == 0:
        return "エラー: 画像を生成してから保存してください", preset_manager.get_gallery_data()

    # 最初の画像を取得
    first_image = gallery_images[0]
    if isinstance(first_image, str):
        # ファイルパスの場合
        first_image = Image.open(first_image)
    elif isinstance(first_image, tuple):
        # (image, caption)のタプルの場合
        if isinstance(first_image[0], str):
            first_image = Image.open(first_image[0])
        else:
            first_image = first_image[0]

    # 設定を辞書にまとめる
    settings = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "enable_hr": enable_hr,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_steps": hr_steps,
        "denoising_strength": denoising_strength,
        "enable_face_restore": enable_face_restore,
        "face_restore_strength": face_restore_strength,
        "adetailer_conf_1": adetailer_conf_1,
        "adetailer_conf_2": adetailer_conf_2,
        "adetailer_conf_3": adetailer_conf_3,
        "adetailer_conf_4": adetailer_conf_4,
    }

    # プリセットを保存
    message = preset_manager.add_preset(preset_name, settings, first_image)

    # 更新されたギャラリーデータを返す
    return message, preset_manager.get_gallery_data()


def load_preset_from_gallery(evt: gr.SelectData) -> Tuple:
    """ギャラリーから選択されたプリセットを読み込み"""

    index = evt.index
    if index >= len(preset_manager.presets):
        return tuple([None] * 24)  # 全ての入力フィールドの数

    preset = preset_manager.presets[index]
    settings = preset["settings"]

    # 設定値を返す（UIコンポーネントの順序に合わせる）
    return (
        settings.get("prompt", ""),
        settings.get("negative_prompt", ""),
        settings.get("sampler_name", "DPM++ 2M"),
        settings.get("scheduler", "Karras"),
        settings.get("steps", 20),
        settings.get("cfg_scale", 7.0),
        settings.get("width", 1024),
        settings.get("height", 1024),
        settings.get("seed", -1),
        settings.get("batch_count", 1),
        settings.get("batch_size", 1),
        settings.get("enable_hr", False),
        settings.get("hr_scale", 2.0),
        settings.get("hr_upscaler", "Latent"),
        settings.get("hr_steps", 20),
        settings.get("denoising_strength", 0.7),
        settings.get("enable_face_restore", False),
        settings.get("face_restore_strength", 0.5),
        settings.get("adetailer_conf_1", 0.3),
        settings.get("adetailer_conf_2", 0.3),
        settings.get("adetailer_conf_3", 0.3),
        settings.get("adetailer_conf_4", 0.3),
        f"プリセット '{preset['name']}' を読み込みました！",
        [],  # プリセット名フィールドをクリア
    )


def refresh_history_gallery():
    """ヒストリーギャラリーを更新"""
    preset_manager.load_presets()
    return preset_manager.get_gallery_data()


def create_custom_diffusion_ui():
    """カスタムディフュージョンUIタブを作成"""

    with gr.Blocks(analytics_enabled=False) as custom_interface:
        gr.Markdown("""
        # 🎨 Custom Diffusion UI

        プリセット管理機能付きの使いやすいカスタムインターフェース

        **固定モデル**: prefectIllustriousXL_v3.safetensors + sdxl_vae.safetensors
        """)

        with gr.Row():
            # 左側：設定パネル
            with gr.Column(scale=1):
                gr.Markdown("### 📝 プロンプト設定")

                prompt = gr.Textbox(
                    label="プロンプト",
                    placeholder="生成したい画像の説明を入力...",
                    lines=3,
                )

                negative_prompt = gr.Textbox(
                    label="ネガティブプロンプト",
                    placeholder="避けたい要素を入力...",
                    lines=2,
                    value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                )

                gr.Markdown("### ⚙️ サンプリング設定")

                with FormRow():
                    sampler_name = gr.Dropdown(
                        label="サンプラー",
                        choices=get_available_samplers(),
                        value="DPM++ 2M",
                    )

                    scheduler = gr.Dropdown(
                        label="スケジューラ",
                        choices=get_available_schedulers(),
                        value="Karras",
                    )

                with FormRow():
                    steps = gr.Slider(
                        label="ステップ数",
                        minimum=1,
                        maximum=150,
                        step=1,
                        value=28,
                    )

                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=30.0,
                        step=0.5,
                        value=7.0,
                    )

                gr.Markdown("### 📐 画像サイズ")

                with FormRow():
                    width = gr.Slider(
                        label="幅",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                    height = gr.Slider(
                        label="高さ",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                gr.Markdown("### 🎲 シード・バッチ")

                with FormRow():
                    seed = gr.Number(
                        label="シード",
                        value=-1,
                        precision=0,
                    )

                    batch_count = gr.Slider(
                        label="バッチ数",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                    )

                    batch_size = gr.Slider(
                        label="バッチサイズ",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=1,
                    )

                gr.Markdown("### 🔍 Hires. fix")

                enable_hr = gr.Checkbox(
                    label="Hires. fix を有効化",
                    value=False,
                )

                with gr.Row(visible=True) as hr_options:
                    with gr.Column():
                        hr_scale = gr.Slider(
                            label="アップスケール倍率",
                            minimum=1.0,
                            maximum=4.0,
                            step=0.1,
                            value=2.0,
                        )

                        hr_upscaler = gr.Dropdown(
                            label="アップスケーラー",
                            choices=get_available_upscalers(),
                            value="Latent",
                        )

                        hr_steps = gr.Slider(
                            label="Hires ステップ数",
                            minimum=0,
                            maximum=150,
                            step=1,
                            value=20,
                        )

                        denoising_strength = gr.Slider(
                            label="Denoising strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.7,
                        )

                gr.Markdown("### 👤 ADetailer設定（フェイスリストア）")

                enable_face_restore = gr.Checkbox(
                    label="フェイスリストアを有効化",
                    value=False,
                )

                with gr.Row(visible=True) as adetailer_options:
                    with gr.Column():
                        face_restore_strength = gr.Slider(
                            label="リストア強度",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.5,
                        )

                        with gr.Row():
                            adetailer_conf_1 = gr.Slider(
                                label="1st 信頼度",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.3,
                            )

                            adetailer_conf_2 = gr.Slider(
                                label="2nd 信頼度",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.3,
                            )

                        with gr.Row():
                            adetailer_conf_3 = gr.Slider(
                                label="3rd 信頼度",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.3,
                            )

                            adetailer_conf_4 = gr.Slider(
                                label="4th 信頼度",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.3,
                            )

                # 生成ボタン
                generate_btn = gr.Button(
                    "🎨 画像を生成",
                    variant="primary",
                    size="lg",
                )

            # 右側：結果パネル
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ 生成結果")

                output_gallery = gr.Gallery(
                    label="生成画像",
                    show_label=False,
                    elem_id="custom_ui_gallery",
                    columns=2,
                    rows=2,
                    height=400,
                )

                generation_info = gr.Textbox(
                    label="生成情報",
                    lines=3,
                    show_copy_button=True,
                )

                html_info = gr.HTML()

                gr.Markdown("### 💾 プリセット管理")

                with gr.Row():
                    preset_name = gr.Textbox(
                        label="プリセット名",
                        placeholder="プリセット名を入力...",
                        scale=3,
                    )

                    save_preset_btn = gr.Button(
                        "💾 保存",
                        scale=1,
                    )

                preset_message = gr.Textbox(
                    label="メッセージ",
                    interactive=False,
                    show_label=False,
                )

                gr.Markdown("### 📚 プリセットヒストリー")
                gr.Markdown("*画像をクリックして設定を読み込み*")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 更新")

                history_gallery = gr.Gallery(
                    label="保存済みプリセット",
                    show_label=False,
                    elem_id="preset_history_gallery",
                    columns=3,
                    rows=3,
                    height=400,
                    value=preset_manager.get_gallery_data(),
                )

        # イベントハンドラを設定

        # 生成ボタンのクリックイベント
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                sampler_name,
                scheduler,
                steps,
                cfg_scale,
                width,
                height,
                seed,
                batch_count,
                batch_size,
                enable_hr,
                hr_scale,
                hr_upscaler,
                hr_steps,
                denoising_strength,
                enable_face_restore,
                face_restore_strength,
                adetailer_conf_1,
                adetailer_conf_2,
                adetailer_conf_3,
                adetailer_conf_4,
            ],
            outputs=[output_gallery, generation_info, html_info],
        )

        # プリセット保存ボタンのクリックイベント
        save_preset_btn.click(
            fn=save_preset,
            inputs=[
                preset_name,
                output_gallery,
                prompt,
                negative_prompt,
                sampler_name,
                scheduler,
                steps,
                cfg_scale,
                width,
                height,
                seed,
                batch_count,
                batch_size,
                enable_hr,
                hr_scale,
                hr_upscaler,
                hr_steps,
                denoising_strength,
                enable_face_restore,
                face_restore_strength,
                adetailer_conf_1,
                adetailer_conf_2,
                adetailer_conf_3,
                adetailer_conf_4,
            ],
            outputs=[preset_message, history_gallery],
        )

        # ヒストリーギャラリーの選択イベント
        history_gallery.select(
            fn=load_preset_from_gallery,
            inputs=[],
            outputs=[
                prompt,
                negative_prompt,
                sampler_name,
                scheduler,
                steps,
                cfg_scale,
                width,
                height,
                seed,
                batch_count,
                batch_size,
                enable_hr,
                hr_scale,
                hr_upscaler,
                hr_steps,
                denoising_strength,
                enable_face_restore,
                face_restore_strength,
                adetailer_conf_1,
                adetailer_conf_2,
                adetailer_conf_3,
                adetailer_conf_4,
                preset_message,
                preset_name,
            ],
        )

        # 更新ボタンのクリックイベント
        refresh_btn.click(
            fn=refresh_history_gallery,
            inputs=[],
            outputs=[history_gallery],
        )

    return custom_interface, "Custom Diffusion", "custom_diffusion"


# タブをUIに登録
script_callbacks.on_ui_tabs(create_custom_diffusion_ui)
