"""
Custom Diffusion UI - プリセット管理機能・キューイング機能付きカスタムインターフェース
"""

import os
import json
import base64
import datetime
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import deque

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


class GenerationQueue:
    """生成キュー管理クラス"""

    def __init__(self):
        self.queue: deque = deque()
        self.results: List[Dict[str, Any]] = []
        self.is_processing = False
        self.current_task_id = 0
        self.lock = threading.Lock()

    def add_task(self, task: Dict[str, Any]) -> int:
        """タスクをキューに追加"""
        with self.lock:
            self.current_task_id += 1
            task['id'] = self.current_task_id
            task['status'] = 'waiting'
            task['added_at'] = datetime.datetime.now().isoformat()
            self.queue.append(task)
            return self.current_task_id

    def get_next_task(self) -> Dict[str, Any]:
        """次のタスクを取得"""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def clear_queue(self):
        """キューをクリア"""
        with self.lock:
            self.queue.clear()

    def get_queue_status(self) -> str:
        """キューの状態を取得"""
        with self.lock:
            if not self.queue:
                status = "キューは空です"
            else:
                status = f"待機中: {len(self.queue)}件\n\n"
                for i, task in enumerate(self.queue, 1):
                    prompt_preview = task['prompt'][:50] + "..." if len(task['prompt']) > 50 else task['prompt']
                    status += f"{i}. [{task['id']}] {prompt_preview}\n"

            if self.is_processing:
                status = "🔄 処理中...\n\n" + status

            return status

    def get_queue_count(self) -> int:
        """キュー内のタスク数を取得"""
        with self.lock:
            return len(self.queue)

    def add_result(self, task_id: int, images: List, info: str):
        """結果を追加"""
        with self.lock:
            self.results.append({
                'task_id': task_id,
                'images': images,
                'info': info,
                'completed_at': datetime.datetime.now().isoformat()
            })

    def get_all_result_images(self) -> List:
        """全結果画像を取得"""
        with self.lock:
            all_images = []
            for result in self.results:
                all_images.extend(result['images'])
            return all_images

    def clear_results(self):
        """結果をクリア"""
        with self.lock:
            self.results.clear()


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


# グローバルインスタンス
preset_manager = PresetManager()
generation_queue = GenerationQueue()


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


def generate_single_image(task: Dict[str, Any]) -> Tuple[List[Image.Image], str, str]:
    """単一タスクの画像を生成"""

    # 固定モデル設定の確認（オーバーライド）
    override_settings = {}

    # 画像生成パラメータを設定
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=task['prompt'],
        negative_prompt=task['negative_prompt'],
        seed=task['seed'],
        sampler_name=task['sampler_name'],
        scheduler=task['scheduler'],
        batch_size=task['batch_size'],
        n_iter=task['batch_count'],
        steps=task['steps'],
        cfg_scale=task['cfg_scale'],
        width=task['width'],
        height=task['height'],
        enable_hr=task['enable_hr'],
        hr_scale=task['hr_scale'],
        hr_upscaler=task['hr_upscaler'],
        hr_second_pass_steps=task['hr_steps'],
        denoising_strength=task['denoising_strength'],
        override_settings=override_settings,
    )

    # フェイスリストア設定
    if task['enable_face_restore']:
        p.restore_faces = True
        p.extra_generation_params["ADetailer Conf 1st"] = task['adetailer_conf_1']
        p.extra_generation_params["ADetailer Conf 2nd"] = task['adetailer_conf_2']
        p.extra_generation_params["ADetailer Conf 3rd"] = task['adetailer_conf_3']
        p.extra_generation_params["ADetailer Conf 4th"] = task['adetailer_conf_4']
        p.extra_generation_params["Face Restore Strength"] = task['face_restore_strength']

    # 画像生成実行
    processed: Processed = process_images(p)

    return processed.images, processed.info, processed.comments_html


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
    enable_face_restore: bool,
    face_restore_strength: float,
    adetailer_conf_1: float,
    adetailer_conf_2: float,
    adetailer_conf_3: float,
    adetailer_conf_4: float,
) -> Tuple[List[Image.Image], str, str]:
    """即時画像生成"""

    task = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'sampler_name': sampler_name,
        'scheduler': scheduler,
        'steps': steps,
        'cfg_scale': cfg_scale,
        'width': width,
        'height': height,
        'seed': seed,
        'batch_count': batch_count,
        'batch_size': batch_size,
        'enable_hr': enable_hr,
        'hr_scale': hr_scale,
        'hr_upscaler': hr_upscaler,
        'hr_steps': hr_steps,
        'denoising_strength': denoising_strength,
        'enable_face_restore': enable_face_restore,
        'face_restore_strength': face_restore_strength,
        'adetailer_conf_1': adetailer_conf_1,
        'adetailer_conf_2': adetailer_conf_2,
        'adetailer_conf_3': adetailer_conf_3,
        'adetailer_conf_4': adetailer_conf_4,
    }

    return generate_single_image(task)


def add_to_queue(
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
) -> Tuple[str, str]:
    """タスクをキューに追加"""

    if not prompt:
        return "エラー: プロンプトを入力してください", generation_queue.get_queue_status()

    task = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'sampler_name': sampler_name,
        'scheduler': scheduler,
        'steps': steps,
        'cfg_scale': cfg_scale,
        'width': width,
        'height': height,
        'seed': seed,
        'batch_count': batch_count,
        'batch_size': batch_size,
        'enable_hr': enable_hr,
        'hr_scale': hr_scale,
        'hr_upscaler': hr_upscaler,
        'hr_steps': hr_steps,
        'denoising_strength': denoising_strength,
        'enable_face_restore': enable_face_restore,
        'face_restore_strength': face_restore_strength,
        'adetailer_conf_1': adetailer_conf_1,
        'adetailer_conf_2': adetailer_conf_2,
        'adetailer_conf_3': adetailer_conf_3,
        'adetailer_conf_4': adetailer_conf_4,
    }

    task_id = generation_queue.add_task(task)
    return f"タスク #{task_id} をキューに追加しました！", generation_queue.get_queue_status()


def process_queue() -> Tuple[List, str, str, str]:
    """キューを処理"""

    if generation_queue.is_processing:
        return [], "エラー: 既に処理中です", "", generation_queue.get_queue_status()

    if generation_queue.get_queue_count() == 0:
        return [], "キューは空です", "", generation_queue.get_queue_status()

    generation_queue.is_processing = True
    generation_queue.clear_results()

    all_images = []
    all_info = []

    try:
        while True:
            task = generation_queue.get_next_task()
            if task is None:
                break

            try:
                images, info, html = generate_single_image(task)
                generation_queue.add_result(task['id'], images, info)
                all_images.extend(images)
                all_info.append(f"[タスク #{task['id']}]\n{info}")
            except Exception as e:
                all_info.append(f"[タスク #{task['id']}] エラー: {str(e)}")

    finally:
        generation_queue.is_processing = False

    combined_info = "\n\n---\n\n".join(all_info)
    message = f"✅ キュー処理完了！{len(all_images)}枚の画像を生成しました"

    return all_images, combined_info, message, generation_queue.get_queue_status()


def clear_queue() -> Tuple[str, str]:
    """キューをクリア"""
    generation_queue.clear_queue()
    return "キューをクリアしました", generation_queue.get_queue_status()


def get_queue_status() -> str:
    """キュー状態を取得"""
    return generation_queue.get_queue_status()


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
        first_image = Image.open(first_image)
    elif isinstance(first_image, tuple):
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

    message = preset_manager.add_preset(preset_name, settings, first_image)
    return message, preset_manager.get_gallery_data()


def load_preset_from_gallery(evt: gr.SelectData) -> Tuple:
    """ギャラリーから選択されたプリセットを読み込み"""

    index = evt.index
    if index >= len(preset_manager.presets):
        return tuple([None] * 24)

    preset = preset_manager.presets[index]
    settings = preset["settings"]

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
        "",
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

        プリセット管理・キューイング機能付きの使いやすいカスタムインターフェース

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
                gr.Markdown("### 🚀 生成")

                with gr.Row():
                    generate_btn = gr.Button(
                        "🎨 即時生成",
                        variant="primary",
                    )

                    add_queue_btn = gr.Button(
                        "📥 キューに追加",
                        variant="secondary",
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

                # キュー管理セクション
                gr.Markdown("### 📋 キュー管理")

                with gr.Row():
                    process_queue_btn = gr.Button(
                        "▶️ キュー実行",
                        variant="primary",
                    )

                    clear_queue_btn = gr.Button(
                        "🗑️ キュークリア",
                        variant="stop",
                    )

                    refresh_queue_btn = gr.Button(
                        "🔄 更新",
                    )

                queue_message = gr.Textbox(
                    label="メッセージ",
                    interactive=False,
                    show_label=False,
                )

                queue_status = gr.Textbox(
                    label="キュー状態",
                    lines=8,
                    interactive=False,
                    value=generation_queue.get_queue_status(),
                )

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

        # 入力コンポーネントのリスト
        generation_inputs = [
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
        ]

        # イベントハンドラを設定

        # 即時生成ボタン
        generate_btn.click(
            fn=generate_image,
            inputs=generation_inputs,
            outputs=[output_gallery, generation_info, html_info],
        )

        # キューに追加ボタン
        add_queue_btn.click(
            fn=add_to_queue,
            inputs=generation_inputs,
            outputs=[queue_message, queue_status],
        )

        # キュー実行ボタン
        process_queue_btn.click(
            fn=process_queue,
            inputs=[],
            outputs=[output_gallery, generation_info, queue_message, queue_status],
        )

        # キュークリアボタン
        clear_queue_btn.click(
            fn=clear_queue,
            inputs=[],
            outputs=[queue_message, queue_status],
        )

        # キュー更新ボタン
        refresh_queue_btn.click(
            fn=get_queue_status,
            inputs=[],
            outputs=[queue_status],
        )

        # プリセット保存ボタン
        save_preset_btn.click(
            fn=save_preset,
            inputs=[
                preset_name,
                output_gallery,
            ] + generation_inputs,
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

        # プリセット更新ボタン
        refresh_btn.click(
            fn=refresh_history_gallery,
            inputs=[],
            outputs=[history_gallery],
        )

    return custom_interface, "Custom Diffusion", "custom_diffusion"


# タブをUIに登録
script_callbacks.on_ui_tabs(create_custom_diffusion_ui)
