import io
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image, ImageOps

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("مرحباً! أرسل صورة وسأحولها إلى أبيض وأسود 🎨")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(image_bytes))
        bw_image = ImageOps.grayscale(image)
        output = io.BytesIO()
        bw_image.save(output, format="JPEG")
        output.seek(0)
        await update.message.reply_photo(photo=output, caption="تم تحويلها! ✨")
    except Exception as e:
        await update.message.reply_text(f"خطأ: {str(e)}")

if __name__ == "__main__":
    TOKEN = "8555031080:AAHKv_xFrNa2POPRPN9eI4TMjXAmEw1XLTw"
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    print("البوت شغّال...")
    app.run_polling()
