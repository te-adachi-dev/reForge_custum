"""
Custom Diffusion UI - プリセット管理機能・キューイング機能付きカスタムインターフェース
"""

import os
import json
import base64
import datetime
import threading
import time
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import deque

import gradio as gr
from PIL import Image

from modules import script_callbacks, shared, sd_samplers, sd_schedulers, scripts
from modules.processing import StableDiffusionProcessingTxt2Img, process_images, Processed
from modules.ui_components import FormRow


# ============================================================================
# ロガー設定
# ============================================================================

def setup_logger():
    """カスタムロガーを設定"""
    logger = logging.getLogger("CustomDiffusionUI")
    logger.setLevel(logging.DEBUG)

    # コンソールハンドラー
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 日本語・時刻付きフォーマット
        formatter = logging.Formatter(
            '[%(asctime)s] [CustomDiffusionUI] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()


def log_separator():
    """ログセパレーター"""
    logger.info("=" * 80)


def log_function_start(func_name: str, **kwargs):
    """関数開始ログ"""
    log_separator()
    logger.info(f"🚀 関数開始: {func_name}")
    if kwargs:
        logger.info(f"   引数:")
        for key, value in kwargs.items():
            # 長い値は省略
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            logger.info(f"      {key}: {str_value}")


def log_function_end(func_name: str, success: bool = True, result_info: str = ""):
    """関数終了ログ"""
    if success:
        logger.info(f"✅ 関数完了: {func_name}")
    else:
        logger.error(f"❌ 関数失敗: {func_name}")
    if result_info:
        logger.info(f"   結果: {result_info}")
    log_separator()


# ============================================================================
# プリセットディレクトリのパス
# ============================================================================

PRESET_DIR = Path(__file__).parent.parent / "presets"
PRESET_FILE = PRESET_DIR / "history.json"
THUMBNAIL_DIR = PRESET_DIR / "thumbnails"

# ディレクトリを作成
logger.info(f"📁 プリセットディレクトリ: {PRESET_DIR}")
logger.info(f"📁 サムネイルディレクトリ: {THUMBNAIL_DIR}")
PRESET_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
logger.info("📁 ディレクトリ作成完了")


# ============================================================================
# キュー管理クラス
# ============================================================================

class GenerationQueue:
    """生成キュー管理クラス"""

    def __init__(self):
        logger.info("🔧 GenerationQueue 初期化開始")
        self.queue: deque = deque()
        self.results: List[Dict[str, Any]] = []
        self.is_processing = False
        self.current_task_id = 0
        self.lock = threading.Lock()
        logger.info("🔧 GenerationQueue 初期化完了")

    def add_task(self, task: Dict[str, Any]) -> int:
        """タスクをキューに追加"""
        with self.lock:
            self.current_task_id += 1
            task['id'] = self.current_task_id
            task['status'] = 'waiting'
            task['added_at'] = datetime.datetime.now().isoformat()
            self.queue.append(task)

            logger.info(f"📥 タスク追加: ID={self.current_task_id}")
            logger.info(f"   プロンプト: {task['prompt'][:50]}...")
            logger.info(f"   キュー内タスク数: {len(self.queue)}")

            return self.current_task_id

    def get_next_task(self) -> Dict[str, Any]:
        """次のタスクを取得"""
        with self.lock:
            if self.queue:
                task = self.queue.popleft()
                logger.info(f"📤 タスク取得: ID={task['id']}")
                logger.info(f"   残りタスク数: {len(self.queue)}")
                return task
            logger.info("📭 キューは空です")
            return None

    def clear_queue(self):
        """キューをクリア"""
        with self.lock:
            count = len(self.queue)
            self.queue.clear()
            logger.info(f"🗑️ キュークリア: {count}件のタスクを削除")

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
            logger.info(f"📊 結果追加: タスクID={task_id}, 画像数={len(images)}")

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
            count = len(self.results)
            self.results.clear()
            logger.info(f"🗑️ 結果クリア: {count}件の結果を削除")


# ============================================================================
# プリセット管理クラス
# ============================================================================

class PresetManager:
    """プリセット管理クラス"""

    def __init__(self):
        logger.info("🔧 PresetManager 初期化開始")
        self.presets: List[Dict[str, Any]] = []
        self.load_presets()
        logger.info(f"🔧 PresetManager 初期化完了: {len(self.presets)}件のプリセットを読み込み")

    def load_presets(self):
        """プリセットをファイルから読み込み"""
        logger.info(f"📂 プリセット読み込み開始: {PRESET_FILE}")
        if PRESET_FILE.exists():
            try:
                with open(PRESET_FILE, 'r', encoding='utf-8') as f:
                    self.presets = json.load(f)
                logger.info(f"✅ プリセット読み込み成功: {len(self.presets)}件")
            except Exception as e:
                logger.error(f"❌ プリセット読み込みエラー: {e}")
                logger.error(traceback.format_exc())
                self.presets = []
        else:
            logger.info("📂 プリセットファイルが存在しません。新規作成します。")
            self.presets = []

    def save_presets(self):
        """プリセットをファイルに保存"""
        logger.info(f"💾 プリセット保存開始: {PRESET_FILE}")
        try:
            with open(PRESET_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ プリセット保存成功: {len(self.presets)}件")
        except Exception as e:
            logger.error(f"❌ プリセット保存エラー: {e}")
            logger.error(traceback.format_exc())

    def add_preset(self, name: str, settings: Dict[str, Any], image: Image.Image) -> str:
        """新しいプリセットを追加"""
        log_function_start("PresetManager.add_preset", name=name)

        try:
            timestamp = datetime.datetime.now().isoformat()
            preset_id = f"preset_{len(self.presets)}_{int(datetime.datetime.now().timestamp())}"

            # サムネイル画像を保存
            thumbnail_path = THUMBNAIL_DIR / f"{preset_id}.png"
            thumbnail = image.copy()
            thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
            thumbnail.save(thumbnail_path)
            logger.info(f"🖼️ サムネイル保存: {thumbnail_path}")

            preset = {
                "id": preset_id,
                "name": name,
                "timestamp": timestamp,
                "thumbnail": str(thumbnail_path),
                "settings": settings
            }

            self.presets.insert(0, preset)  # 最新のものを先頭に
            self.save_presets()

            message = f"プリセット '{name}' を保存しました！"
            log_function_end("PresetManager.add_preset", success=True, result_info=message)
            return message

        except Exception as e:
            error_msg = f"プリセット保存エラー: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            log_function_end("PresetManager.add_preset", success=False, result_info=str(e))
            return f"エラー: {e}"

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
        logger.info(f"🖼️ ギャラリーデータ取得: {len(gallery_data)}件")
        return gallery_data


# ============================================================================
# グローバルインスタンス
# ============================================================================

logger.info("🌐 グローバルインスタンス作成開始")
preset_manager = PresetManager()
generation_queue = GenerationQueue()
logger.info("🌐 グローバルインスタンス作成完了")


# ============================================================================
# ユーティリティ関数
# ============================================================================

def get_available_samplers() -> List[str]:
    """利用可能なサンプラーのリストを取得"""
    samplers = ["DPM++ 2M", "DPM++ 3M SDE", "Euler a"]
    logger.debug(f"📋 利用可能なサンプラー: {samplers}")
    return samplers


def get_available_schedulers() -> List[str]:
    """利用可能なスケジューラーのリストを取得"""
    schedulers = ["Karras", "SGM Uniform"]
    logger.debug(f"📋 利用可能なスケジューラー: {schedulers}")
    return schedulers


def get_available_upscalers() -> List[str]:
    """利用可能なアップスケーラーのリストを取得"""
    log_function_start("get_available_upscalers")

    upscalers = ["None"]

    try:
        # 全てのアップスケーラーから指定されたものを探す
        logger.info(f"🔍 登録されているアップスケーラー数: {len(shared.sd_upscalers)}")

        for upscaler in shared.sd_upscalers:
            name = upscaler.name
            logger.debug(f"   アップスケーラー検出: {name}")

            # 要求されたアップスケーラーを探す
            if "4x-UltraSharp" in name or "UltraSharp" in name:
                upscalers.append(name)
                logger.info(f"   ✅ 追加: {name}")
            elif "DAT" in name and "x4" in name:
                upscalers.append(name)
                logger.info(f"   ✅ 追加: {name}")
            elif "ESRGAN" in name and "4x" in name and "Anime6B" in name:
                upscalers.append(name)
                logger.info(f"   ✅ 追加: {name}")

        # デフォルトのアップスケーラーも追加
        if "Latent" not in upscalers:
            upscalers.append("Latent")
        if "DAT x4" not in upscalers:
            upscalers.append("DAT x4")

        logger.info(f"📋 最終アップスケーラーリスト: {upscalers}")
        log_function_end("get_available_upscalers", success=True, result_info=f"{len(upscalers)}件")

    except Exception as e:
        logger.error(f"❌ アップスケーラー取得エラー: {e}")
        logger.error(traceback.format_exc())
        log_function_end("get_available_upscalers", success=False)

    return upscalers


# ============================================================================
# 画像生成関数
# ============================================================================

def generate_single_image(task: Dict[str, Any]) -> Tuple[List[Image.Image], str, str]:
    """単一タスクの画像を生成"""

    log_function_start("generate_single_image",
        prompt=task['prompt'][:50] + "...",
        sampler=task['sampler_name'],
        scheduler=task['scheduler'],
        steps=task['steps'],
        cfg_scale=task['cfg_scale'],
        width=task['width'],
        height=task['height'],
        seed=task['seed'],
        enable_hr=task['enable_hr']
    )

    try:
        # 固定モデル設定の確認（オーバーライド）
        override_settings = {}

        # 現在のモデル情報をログ
        logger.info(f"🤖 現在のモデル: {shared.sd_model.sd_checkpoint_info.title if shared.sd_model else 'None'}")
        logger.info(f"📁 出力ディレクトリ (samples): {shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples}")
        logger.info(f"📁 出力ディレクトリ (grids): {shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids}")

        # 画像生成パラメータを設定
        logger.info("⚙️ StableDiffusionProcessingTxt2Img オブジェクト作成開始")

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

        logger.info("✅ StableDiffusionProcessingTxt2Img オブジェクト作成完了")

        # フェイスリストア設定
        if task['enable_face_restore']:
            logger.info("👤 フェイスリストア設定を適用")
            p.restore_faces = True
            p.extra_generation_params["ADetailer Conf 1st"] = task['adetailer_conf_1']
            p.extra_generation_params["ADetailer Conf 2nd"] = task['adetailer_conf_2']
            p.extra_generation_params["ADetailer Conf 3rd"] = task['adetailer_conf_3']
            p.extra_generation_params["ADetailer Conf 4th"] = task['adetailer_conf_4']
            p.extra_generation_params["Face Restore Strength"] = task['face_restore_strength']
            logger.info(f"   リストア強度: {task['face_restore_strength']}")
            logger.info(f"   信頼度: {task['adetailer_conf_1']}, {task['adetailer_conf_2']}, {task['adetailer_conf_3']}, {task['adetailer_conf_4']}")

        # スクリプトの初期化
        logger.info("📜 スクリプト初期化開始")
        logger.info(f"   scripts.scripts_txt2img: {scripts.scripts_txt2img}")
        logger.info(f"   scripts.scripts_txt2img の型: {type(scripts.scripts_txt2img)}")

        p.scripts = scripts.scripts_txt2img
        p.script_args = []

        logger.info("✅ スクリプト初期化完了")
        logger.info(f"   p.scripts: {p.scripts}")
        logger.info(f"   p.script_args: {p.script_args}")

        # 画像生成実行
        logger.info("🎨 画像生成開始...")
        start_time = time.time()

        processed: Processed = process_images(p)

        elapsed_time = time.time() - start_time
        logger.info(f"✅ 画像生成完了: {elapsed_time:.2f}秒")
        logger.info(f"   生成画像数: {len(processed.images)}")
        logger.info(f"   シード: {processed.seed}")

        log_function_end("generate_single_image", success=True,
                        result_info=f"{len(processed.images)}枚生成, {elapsed_time:.2f}秒")

        return processed.images, processed.info, processed.comments_html

    except Exception as e:
        error_msg = f"画像生成エラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"📋 スタックトレース:\n{traceback.format_exc()}")
        log_function_end("generate_single_image", success=False, result_info=str(e))
        raise


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

    log_function_start("generate_image (即時生成)", prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt)

    try:
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

        result = generate_single_image(task)
        log_function_end("generate_image (即時生成)", success=True)
        return result

    except Exception as e:
        error_msg = f"即時生成エラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        log_function_end("generate_image (即時生成)", success=False, result_info=str(e))
        return [], f"エラー: {e}", ""


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

    log_function_start("add_to_queue", prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt)

    try:
        if not prompt:
            logger.warning("⚠️ プロンプトが空です")
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
        message = f"タスク #{task_id} をキューに追加しました！"

        log_function_end("add_to_queue", success=True, result_info=message)
        return message, generation_queue.get_queue_status()

    except Exception as e:
        error_msg = f"キュー追加エラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        log_function_end("add_to_queue", success=False, result_info=str(e))
        return f"エラー: {e}", generation_queue.get_queue_status()


def process_queue() -> Tuple[List, str, str, str]:
    """キューを処理"""

    log_function_start("process_queue")

    if generation_queue.is_processing:
        logger.warning("⚠️ 既に処理中です")
        return [], "エラー: 既に処理中です", "", generation_queue.get_queue_status()

    queue_count = generation_queue.get_queue_count()
    if queue_count == 0:
        logger.warning("⚠️ キューは空です")
        return [], "キューは空です", "", generation_queue.get_queue_status()

    logger.info(f"📋 キュー処理開始: {queue_count}件のタスク")
    generation_queue.is_processing = True
    generation_queue.clear_results()

    all_images = []
    all_info = []
    success_count = 0
    error_count = 0

    try:
        task_number = 0
        while True:
            task = generation_queue.get_next_task()
            if task is None:
                logger.info("📭 キューが空になりました")
                break

            task_number += 1
            logger.info(f"🔄 タスク処理中: {task_number}/{queue_count} (ID: {task['id']})")

            try:
                images, info, html = generate_single_image(task)
                generation_queue.add_result(task['id'], images, info)
                all_images.extend(images)
                all_info.append(f"[タスク #{task['id']}]\n{info}")
                success_count += 1
                logger.info(f"✅ タスク #{task['id']} 完了: {len(images)}枚生成")

            except Exception as e:
                error_msg = f"[タスク #{task['id']}] エラー: {str(e)}"
                all_info.append(error_msg)
                error_count += 1
                logger.error(f"❌ {error_msg}")
                logger.error(traceback.format_exc())

    finally:
        generation_queue.is_processing = False
        logger.info("🏁 キュー処理完了")

    combined_info = "\n\n---\n\n".join(all_info)
    message = f"✅ キュー処理完了！{len(all_images)}枚の画像を生成しました (成功: {success_count}, エラー: {error_count})"

    log_function_end("process_queue", success=True, result_info=message)

    return all_images, combined_info, message, generation_queue.get_queue_status()


def clear_queue() -> Tuple[str, str]:
    """キューをクリア"""
    log_function_start("clear_queue")
    generation_queue.clear_queue()
    message = "キューをクリアしました"
    log_function_end("clear_queue", success=True, result_info=message)
    return message, generation_queue.get_queue_status()


def get_queue_status() -> str:
    """キュー状態を取得"""
    return generation_queue.get_queue_status()


# ============================================================================
# プリセット管理関数
# ============================================================================

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

    log_function_start("save_preset", preset_name=preset_name)

    try:
        if not preset_name:
            logger.warning("⚠️ プリセット名が空です")
            return "エラー: プリセット名を入力してください", preset_manager.get_gallery_data()

        if not gallery_images or len(gallery_images) == 0:
            logger.warning("⚠️ 画像がありません")
            return "エラー: 画像を生成してから保存してください", preset_manager.get_gallery_data()

        logger.info(f"🖼️ ギャラリー画像数: {len(gallery_images)}")
        logger.info(f"🖼️ ギャラリー画像タイプ: {type(gallery_images[0])}")

        # 最初の画像を取得
        first_image = gallery_images[0]
        if isinstance(first_image, str):
            logger.info(f"📂 画像パスから読み込み: {first_image}")
            first_image = Image.open(first_image)
        elif isinstance(first_image, tuple):
            if isinstance(first_image[0], str):
                logger.info(f"📂 タプルから画像パスを取得: {first_image[0]}")
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
        log_function_end("save_preset", success=True, result_info=message)
        return message, preset_manager.get_gallery_data()

    except Exception as e:
        error_msg = f"プリセット保存エラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        log_function_end("save_preset", success=False, result_info=str(e))
        return f"エラー: {e}", preset_manager.get_gallery_data()


def load_preset_from_gallery(evt: gr.SelectData) -> Tuple:
    """ギャラリーから選択されたプリセットを読み込み"""

    log_function_start("load_preset_from_gallery", index=evt.index)

    try:
        index = evt.index
        if index >= len(preset_manager.presets):
            logger.warning(f"⚠️ インデックスが範囲外: {index} >= {len(preset_manager.presets)}")
            return tuple([None] * 24)

        preset = preset_manager.presets[index]
        settings = preset["settings"]

        logger.info(f"📂 プリセット読み込み: {preset['name']}")
        logger.info(f"   ID: {preset['id']}")
        logger.info(f"   作成日時: {preset['timestamp']}")

        result = (
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

        log_function_end("load_preset_from_gallery", success=True, result_info=f"プリセット '{preset['name']}' を読み込み")
        return result

    except Exception as e:
        error_msg = f"プリセット読み込みエラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        log_function_end("load_preset_from_gallery", success=False, result_info=str(e))
        return tuple([None] * 24)


def refresh_history_gallery():
    """ヒストリーギャラリーを更新"""
    log_function_start("refresh_history_gallery")
    preset_manager.load_presets()
    result = preset_manager.get_gallery_data()
    log_function_end("refresh_history_gallery", success=True, result_info=f"{len(result)}件")
    return result


# ============================================================================
# UI作成関数
# ============================================================================

def create_custom_diffusion_ui():
    """カスタムディフュージョンUIタブを作成"""

    log_function_start("create_custom_diffusion_ui")

    try:
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
            logger.info("🔗 イベントハンドラ設定開始")

            # 即時生成ボタン
            generate_btn.click(
                fn=generate_image,
                inputs=generation_inputs,
                outputs=[output_gallery, generation_info, html_info],
            )
            logger.info("   ✅ 即時生成ボタン設定完了")

            # キューに追加ボタン
            add_queue_btn.click(
                fn=add_to_queue,
                inputs=generation_inputs,
                outputs=[queue_message, queue_status],
            )
            logger.info("   ✅ キュー追加ボタン設定完了")

            # キュー実行ボタン
            process_queue_btn.click(
                fn=process_queue,
                inputs=[],
                outputs=[output_gallery, generation_info, queue_message, queue_status],
            )
            logger.info("   ✅ キュー実行ボタン設定完了")

            # キュークリアボタン
            clear_queue_btn.click(
                fn=clear_queue,
                inputs=[],
                outputs=[queue_message, queue_status],
            )
            logger.info("   ✅ キュークリアボタン設定完了")

            # キュー更新ボタン
            refresh_queue_btn.click(
                fn=get_queue_status,
                inputs=[],
                outputs=[queue_status],
            )
            logger.info("   ✅ キュー更新ボタン設定完了")

            # プリセット保存ボタン
            save_preset_btn.click(
                fn=save_preset,
                inputs=[
                    preset_name,
                    output_gallery,
                ] + generation_inputs,
                outputs=[preset_message, history_gallery],
            )
            logger.info("   ✅ プリセット保存ボタン設定完了")

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
            logger.info("   ✅ ヒストリーギャラリー選択イベント設定完了")

            # プリセット更新ボタン
            refresh_btn.click(
                fn=refresh_history_gallery,
                inputs=[],
                outputs=[history_gallery],
            )
            logger.info("   ✅ プリセット更新ボタン設定完了")

        logger.info("🔗 イベントハンドラ設定完了")
        log_function_end("create_custom_diffusion_ui", success=True)

        return [(custom_interface, "Custom Diffusion", "custom_diffusion")]

    except Exception as e:
        error_msg = f"UI作成エラー: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        log_function_end("create_custom_diffusion_ui", success=False, result_info=str(e))
        raise


# ============================================================================
# タブをUIに登録
# ============================================================================

logger.info("🎯 Custom Diffusion UI 拡張機能を登録中...")
script_callbacks.on_ui_tabs(create_custom_diffusion_ui)
logger.info("✅ Custom Diffusion UI 拡張機能の登録完了")
