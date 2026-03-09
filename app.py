import streamlit as st
import anthropic, fal_client, requests, base64, os, time, tempfile, shutil
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
from fpdf import FPDF

st.set_page_config(page_title="Awwa Stories – Demo", page_icon="📖", layout="centered")

FAL_KEY = st.secrets.get("FAL_KEY", os.environ.get("FAL_KEY", ""))
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
os.environ["FAL_KEY"] = FAL_KEY

# デバッグ表示（後で消す）
st.sidebar.write("FAL_KEY set:", bool(FAL_KEY))
st.sidebar.write("ANTHROPIC_KEY set:", bool(ANTHROPIC_API_KEY))
st.sidebar.write("FAL_KEY prefix:", FAL_KEY[:8] if FAL_KEY else "EMPTY")

BASE_DIR = Path(__file__).parent
SCENES = [
    ("scene_01", BASE_DIR / "illustrations" / "scene_01.jpg"),
    ("scene_02", BASE_DIR / "illustrations" / "scene_02.jpg"),
    ("scene_03", BASE_DIR / "illustrations" / "scene_03.jpg"),
    ("scene_04", BASE_DIR / "illustrations" / "scene_04.jpg"),
    ("scene_05", BASE_DIR / "illustrations" / "scene_05.jpg"),
]
CAPTIONS = [
    "Today, Sara starts her adventure.",
    "She drives her tractor across the farm.",
    "Suddenly, she finds a dinosaur in the forest.",
    "She explores the jungle.",
    "Finally, Sara returns home to her family.",
]

def to_b64(filepath, mime="image/jpeg"):
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def _is_black(img):
    arr = np.array(img.convert("L"), dtype=float)
    return (arr.mean() / 255.0) < 0.02

def analyze_child_photo(photo_bytes, ext):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    photo_b64 = base64.b64encode(photo_bytes).decode()
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    analysis = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": photo_b64}},
            {"type": "text", "text": (
                "Describe this child for an anime picture book illustration. "
                "Output exactly two things separated by '|': "
                "1) One sentence: boy or girl, hair color/style, eye color, age "
                "2) Gender word only: 'girl' or 'boy' "
                'Example: "a 5-year-old girl with long dark brown hair and dark brown eyes|girl" '
                "No extra text."
            )}
        ]}]
    )
    raw = analysis.content[0].text.strip().strip('"')
    if '|' in raw:
        desc, gender = raw.split('|', 1)
        return desc.strip(), gender.strip()
    return raw, 'girl' if 'girl' in raw.lower() else 'boy'

def _build_prompt(scene_id, child_desc, gender):
    pronoun = "his" if gender == "boy" else "her"
    prompts = {
        "scene_01": (
            f"This is an anime-style picture book illustration. "
            f"Find the child character riding a vehicle on the RIGHT half of the image. "
            f"Replace only that character's face and hair to look like: {child_desc}. "
            f"Keep the character's clothing, body, and pose completely unchanged. "
            f"Keep the entire LEFT half and all background elements completely unchanged. "
            f"Do NOT add any text, words, or letters anywhere in the image. "
            f"Preserve the soft anime picture book art style."
        ),
        "scene_02": (
            f"This is an anime-style picture book illustration of a sunny farm with wheat fields. "
            f"There are TWO child characters: a small one on the LEFT on a small yellow tractor, "
            f"and a large one on the RIGHT edge leaning out of a bigger yellow vehicle. "
            f"Change ONLY the face and hair of the large child on the RIGHT "
            f"to look like: {child_desc}. Keep {pronoun} clothing and body unchanged. "
            f"The small child on the LEFT must remain completely unchanged. "
            f"Keep the wheat, trees, orange cat, sky, and all background completely unchanged. "
            f"Do NOT add any text to the image. Preserve the anime picture book art style."
        ),
        "scene_03": (
            f"This is an anime-style picture book illustration of a colorful forest. "
            f"There is a child character sitting on top of a yellow bulldozer in the LEFT half of the image, holding a pink toy. "
            f"An orange cat is also on the bulldozer. "
            f"Replace only the child's face and hair to look like: {child_desc}. "
            f"Keep the child's clothing, body, pose, and the pink toy completely unchanged. "
            f"Keep the orange cat completely unchanged. "
            f"Keep the forest trees, green grass, dinosaurs, river, volcano, and entire RIGHT half completely unchanged. "
            f"Do NOT add any text or words to the image. "
            f"Preserve the anime picture book art style."
        ),
        "scene_04": (
            f"This is an anime-style picture book illustration of a jungle. "
            f"Find the small child character on the LEFT driving a yellow roller machine. "
            f"Replace only that character's face and hair to look like: {child_desc}. "
            f"Keep the character's clothing and body completely unchanged. "
            f"Keep the rocks, blue dinosaur, waterfall, and entire RIGHT half completely unchanged. "
            f"Do NOT add any text or words to the image. "
            f"Preserve the anime picture book art style."
        ),
        "scene_05_left": (
            f"This is a picture book illustration with TWO side-by-side pages. "
            f"The LEFT page shows: a farm scene with a barn, yellow tractor, chickens, and a small standing child. "
            f"The RIGHT page shows: inside a blue car with a bearded man driving, a child in center, woman in purple hijab on right. "
            f"Task: Change ONLY the small standing child on the LEFT farm page "
            f"to look like: {child_desc}. Keep {pronoun} clothing and pose unchanged. "
            f"The RIGHT page with the blue car and all people inside must be pixel-perfect identical to the original. "
            f"The farm background, barn, tractor, chickens, fence on the LEFT must stay unchanged. "
            f"Do NOT add any text to the image. Preserve anime picture book art style."
        ),
        "scene_05_car": (
            f"This illustration has TWO pages side by side. "
            f"LEFT page shows a farm scene. RIGHT page shows inside a blue car. "
            f"The car has: a bearded adult man driving on the LEFT seat, "
            f"a small child in the CENTER seat, and a woman in purple hijab on the RIGHT seat. "
            f"Change ONLY the small child in the CENTER seat "
            f"to look like: {child_desc}. "
            f"The bearded adult male driver on the LEFT must stay completely unchanged. "
            f"The woman in purple hijab on the RIGHT must stay completely unchanged. "
            f"The LEFT page farm scene must stay completely unchanged. "
            f"Do NOT add any text to the image. Preserve anime picture book art style."
        ),
    }
    return prompts.get(scene_id, f"Change the child character to look like: {child_desc}. Keep everything else unchanged. Do not add text.")

def run_kontext(image_path, prompt, out_path, max_retries=3):
    st.write(f"  🔄 Calling fal.ai for {Path(image_path).name}...")
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(2)
        try:
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext",
                arguments={
                    "image_url": to_b64(image_path),
                    "prompt": prompt,
                    "guidance_scale": 5.0,
                    "num_inference_steps": 20,
                    "num_images": 1,
                    "safety_tolerance": "5",
                },
                with_logs=False,
            )
            url = result["images"][0]["url"]
            img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
            if _is_black(img):
                continue
            img.save(out_path, "JPEG", quality=75)
            st.write(f"  ✅ Done: {Path(out_path).name}")
            return out_path
        except Exception as e:
            st.error(f"  ❌ fal.ai error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                shutil.copy(image_path, out_path)
                return out_path
    shutil.copy(image_path, out_path)
    return out_path

def generate_all_scenes(child_desc, gender, tmp_dir):
    for i, (scene_id, base_path) in enumerate(SCENES):
        out_path = str(Path(tmp_dir) / f"{scene_id}.jpg")
        if scene_id == "scene_05":
            tmp5 = str(Path(tmp_dir) / "scene_05_tmp.jpg")
            run_kontext(str(base_path), _build_prompt("scene_05_left", child_desc, gender), tmp5)
            run_kontext(tmp5, _build_prompt("scene_05_car", child_desc, gender), out_path)
            if os.path.exists(tmp5):
                os.remove(tmp5)
        else:
            run_kontext(str(base_path), _build_prompt(scene_id, child_desc, gender), out_path)
        yield i + 1, out_path

def build_pdf(image_paths, captions):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    for img_path, caption in zip(image_paths, captions):
        pdf.add_page()
        # 画像を圧縮してから追加
        img = Image.open(img_path)
        img = img.resize((1200, 675), Image.LANCZOS)
        compressed = img_path.replace(".jpg", "_small.jpg")
        img.save(compressed, "JPEG", quality=60)
        pdf.image(compressed, x=10, y=10, w=277)
        pdf.set_xy(10, 185)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(277, 10, caption, align="C")
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ── UI ───────────────────────────────────────────────────────
st.title("📖 Awwa Stories – Personalized Picture Book")
st.markdown("Upload a child's photo to create a personalized adventure story!")

uploaded = st.file_uploader("Upload child's photo", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded, caption="Uploaded photo", use_container_width=True)
    with col2:
        st.markdown("**Ready to generate your personalized story!**")
        st.markdown("- 5 story scenes\n- ~60 seconds to generate\n- PDF export included")

    if st.button("✨ Generate My Story", type="primary"):
        photo_bytes = uploaded.read()
        ext = Path(uploaded.name).suffix

        with st.spinner("Analyzing photo..."):
            try:
                child_desc, gender = analyze_child_photo(photo_bytes, ext)
                st.success(f"✅ Child detected: {child_desc}")
            except Exception as e:
                st.error(f"❌ Claude API error: {e}")
                st.stop()

        tmp_dir = tempfile.mkdtemp()
        photo_path = os.path.join(tmp_dir, f"photo{ext}")
        with open(photo_path, "wb") as f:
            f.write(photo_bytes)

        st.markdown("### Generating scenes...")
        progress = st.progress(0)
        scene_imgs = []
        previews = st.columns(5)

        for completed, img_path in generate_all_scenes(child_desc, gender, tmp_dir):
            progress.progress(completed / 5)
            with previews[completed - 1]:
                st.image(img_path, use_container_width=True)
                st.caption(CAPTIONS[completed - 1][:30] + "...")
            scene_imgs.append(img_path)

        st.success("🎉 Your story is ready!")

        st.markdown("---")
        st.markdown("### 📖 Your Personalized Story")
        page = st.slider("Page", 1, 5, 1)
        st.image(scene_imgs[page - 1], use_container_width=True)
        st.markdown(f"**{CAPTIONS[page - 1]}**")

        st.markdown("---")
        with st.spinner("Building PDF..."):
            pdf_bytes = build_pdf(scene_imgs, CAPTIONS)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name="awwa_story_personalized.pdf",
            mime="application/pdf",
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)
