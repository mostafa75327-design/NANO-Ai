import io
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForInpainting
import cv2

# تحميل النماذج مجاناً (مرة واحدة فقط)
print("⏳ جاري تحميل النماذج المجانية...")

# نموذج تحرير الصور المتقدم
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    print("✅ تم تحميل نموذج تحرير الصور")
except:
    pipe = None
    print("⚠️ لم يتم تحميل نموذج Stable Diffusion")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 **بوت تعديل الصور المتقدم (مجاني للأبد)**\n\n"
        "📸 **الطريقة:**\n"
        "1️⃣ أرسل صورة\n"
        "2️⃣ اختر التعديل من الأوامر\n\n"
        "**التعديلات المتقدمة:**\n"
        "/enhance - تحسين الجودة والوضوح\n"
        "/artistic - تأثير فني\n"
        "/vivid - ألوان حيّة\n"
        "/neon - تأثير نيون\n"
        "/dreamy - تأثير حلمي\n"
        "/cinematic - تأثير سينمائي\n"
        "/vintage - مظهر قديم\n"
        "/watercolor - ألوان مائية\n"
        "/oil - رسم بالزيت\n"
        "/sketch - رسم بقلم\n"
        "/cartoon - كرتون\n"
        "/sepia - سيبيا\n"
        "/noir - أبيض وأسود احترافي\n"
        "/thermal - تأثير حراري\n"
        "/psychedelic - تأثير نفسي\n\n"
        "**التعديلات الأساسية:**\n"
        "/brightness [1-3] - السطوع\n"
        "/contrast [1-3] - التباين\n"
        "/saturation [1-3] - التشبع\n"
        "/blur - تمويه\n"
        "/sharpen - وضوح\n"
        "/upscale - تحسين الجودة\n"
        "/denoise - إزالة الضوضاء\n"
        "/detail - إضافة تفاصيل\n"
        "/smooth - تنعيم احترافي\n"
        "/edge - كشف الحواف\n"
        "/emboss - نقش\n"
        "/invert - عكس الألوان\n"
        "/rotate - تدوير\n"
        "/flip - قلب\n"
        "/grayscale - رمادي\n\n"
        "⚡ **مجاني للأبد - بدون حدود!**"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        
        # تحويل إلى PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # تحويل إلى OpenCV
        cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # حفظ في context
        context.user_data['pil_image'] = pil_image
        context.user_data['cv_image'] = cv_image
        
        await update.message.reply_text("✅ تم حفظ الصورة! اختر التعديل")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

# ========== التعديلات المتقدمة ==========

async def enhance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        
        # تحسين الجودة
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ تم تحسين الجودة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def artistic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # تأثير فني باستخدام Bilateral Filter
        artistic = cv2.bilateralFilter(image, 9, 75, 75)
        artistic = cv2.bilateralFilter(artistic, 9, 75, 75)
        
        _, buffer = cv2.imencode('.jpg', artistic)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير فني مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def vivid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ ألوان حيّة مضافة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def neon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # تأثير نيون
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        neon = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        _, buffer = cv2.imencode('.jpg', neon)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير نيون مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def dreamy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        
        # تأثير حلمي
        blurred = image.filter(ImageFilter.GaussianBlur(radius=5))
        
        enhancer = ImageEnhance.Brightness(blurred)
        dreamy = enhancer.enhance(1.2)
        
        output = io.BytesIO()
        dreamy.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير حلمي مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def cinematic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        
        # تأثير سينمائي
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.4)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.8)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير سينمائي مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def vintage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.7)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ مظهر قديم مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def watercolor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # تأثير ألوان مائية
        for _ in range(2):
            image = cv2.bilateralFilter(image, 9, 75, 75)
        
        _, buffer = cv2.imencode('.jpg', image)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ ألوان مائية مضافة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def oil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # تأثير رسم بالزيت
        oil = cv2.xphoto.oilPainting(image, 7, 1)
        
        _, buffer = cv2.imencode('.jpg', oil)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ رسم بالزيت مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def sketch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur + 1) * 255
        sketch = sketch.astype('uint8')
        
        _, buffer = cv2.imencode('.jpg', sketch)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ رسم بقلم مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def cartoon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # تأثير كرتون
        image = cv2.bilateralFilter(image, 9, 75, 75)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(image, edges)
        
        _, buffer = cv2.imencode('.jpg', cartoon)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ كرتون مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def sepia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, kernel)
        sepia = np.clip(sepia, 0, 255).astype('uint8')
        
        _, buffer = cv2.imencode('.jpg', sepia)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ سيبيا مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def noir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image'].convert('L')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        image = image.convert('RGB')
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ أبيض وأسود احترافي!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def thermal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        _, buffer = cv2.imencode('.jpg', thermal)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير حراري مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def psychedelic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        h = cv2.add(h, 50)
        s = cv2.multiply(s, 1.5)
        v = cv2.multiply(v, 1.3)
        
        psych = cv2.merge([h, np.clip(s, 0, 255).astype('uint8'), np.clip(v, 0, 255).astype('uint8')])
        psych = cv2.cvtColor(psych, cv2.COLOR_HSV2BGR)
        
        _, buffer = cv2.imencode('.jpg', psych)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تأثير نفسي مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

# ========== التعديلات الأساسية ==========

async def brightness(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        factor = float(context.args[0]) if context.args else 1.5
        image = context.user_data['pil_image']
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ السطوع تعديل!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def contrast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        factor = float(context.args[0]) if context.args else 1.5
        image = context.user_data['pil_image']
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ التباين تعديل!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def saturation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        factor = float(context.args[0]) if context.args else 1.5
        image = context.user_data['pil_image']
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factor)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ التشبع تعديل!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def blur(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        image = image.filter(ImageFilter.GaussianBlur(radius=10))
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ تمويه مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def sharpen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        image = image.filter(ImageFilter.SHARPEN)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ وضوح مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def upscale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image']
        w, h = image.size
        image = image.resize((w*2, h*2), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ تحسين الجودة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def denoise(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 15, 21)
        
        _, buffer = cv2.imencode('.jpg', denoised)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ إزالة الضوضاء!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def detail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        
        # شحذ متقدم
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        
        detailed = cv2.filter2D(image, -1, kernel)
        detailed = np.clip(detailed, 0, 255).astype('uint8')
        
        _, buffer = cv2.imencode('.jpg', detailed)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تفاصيل مضافة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def smooth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        
        _, buffer = cv2.imencode('.jpg', smooth)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تنعيم مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def edge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        _, buffer = cv2.imencode('.jpg', edges)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ كشف الحواف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def emboss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [0,  1, 2]])
        
        embossed = cv2.filter2D(image, -1, kernel)
        embossed = np.clip(embossed, 0, 255).astype('uint8')
        
        _, buffer = cv2.imencode('.jpg', embossed)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ نقش مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def invert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        inverted = cv2.bitwise_not(image)
        
        _, buffer = cv2.imencode('.jpg', inverted)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ ألوان معكوسة!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def rotate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        
        _, buffer = cv2.imencode('.jpg', rotated)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ تدوير مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def flip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'cv_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['cv_image'].copy()
        flipped = cv2.flip(image, 1)
        
        _, buffer = cv2.imencode('.jpg', flipped)
        output = io.BytesIO(buffer)
        
        await update.message.reply_photo(photo=output, caption="✅ قلب مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

async def grayscale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'pil_image' not in context.user_data:
            await update.message.reply_text("❌ أرسل صورة أولاً!")
            return
        
        image = context.user_data['pil_image'].convert('L').convert('RGB')
        
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        await update.message.reply_photo(photo=output, caption="✅ رمادي مضاف!")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {str(e)}")

if __name__ == "__main__":
    TOKEN = "8555031080:AAHKv_xFrNa2POPRPN9eI4TMjXAmEw1XLTw"
    
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # متقدمة
    app.add_handler(CommandHandler("enhance", enhance))
    app.add_handler(CommandHandler("artistic", artistic))
    app.add_handler(CommandHandler("vivid", vivid))
    app.add_handler(CommandHandler("neon", neon))
    app.add_handler(CommandHandler("dreamy", dreamy))
    app.add_handler(CommandHandler("cinematic", cinematic))
    app.add_handler(CommandHandler("vintage", vintage))
    app.add_handler(CommandHandler("watercolor", watercolor))
    app.add_handler(CommandHandler("oil", oil))
    app.add_handler(CommandHandler("sketch", sketch))
    app.add_handler(CommandHandler("cartoon", cartoon))
    app.add_handler(CommandHandler("sepia", sepia))
    app.add_handler(CommandHandler("noir", noir))
    app.add_handler(CommandHandler("thermal", thermal))
    app.add_handler(CommandHandler("psychedelic", psychedelic))
    
    # أساسية
    app.add_handler(CommandHandler("brightness", brightness))
    app.add_handler(CommandHandler("contrast", contrast))
    app.add_handler(CommandHandler("saturation", saturation))
    app.add_handler(CommandHandler("blur", blur))
    app.add_handler(CommandHandler("sharpen", sharpen))
    app.add_handler(CommandHandler("upscale", upscale))
    app.add_handler(CommandHandler("denoise", denoise))
    app.add_handler(CommandHandler("detail", detail))
    app.add_handler(CommandHandler("smooth", smooth))
    app.add_handler(CommandHandler("edge", edge))
    app.add_handler(CommandHandler("emboss", emboss))
    app.add_handler(CommandHandler("invert", invert))
    app.add_handler(CommandHandler("rotate", rotate))
    app.add_handler(CommandHandler("flip", flip))
    app.add_handler(CommandHandler("grayscale", grayscale))
    
    print("🚀 البوت يعمل على Railway - كل التعديلات مجانية للأبد!")
    app.run_polling()
