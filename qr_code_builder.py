import qrcode
from PIL import Image, ImageDraw, ImageFont

def qr_builder(args):
    url = args.url
    title = args.title

    # Create QR code instance with a higher resolution
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=20,  # Increase box size for higher resolution
        border=4,
    )

    # Add data to the QR code
    qr.add_data(url)  # Only add the URL, not the title
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    # Define a font
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the size of the text
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), title, font=font)  # Get bounding box of the title text
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Create a new image with extra space for the title
    new_img = Image.new('RGB', (img.size[0], img.size[1] + text_height + 10), 'white')
    new_img.paste(img, (0, text_height + 10))

    # Draw the title
    draw = ImageDraw.Draw(new_img)
    text_position = ((new_img.size[0] - text_width) // 2, 5)
    draw.text(text_position, title, fill="black", font=font)

    # Save the image with a high DPI setting
    new_img.save(f"{title}_qr_code.png", dpi=(300, 300))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a QR code with a title')
    parser.add_argument('--url', type=str)
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    qr_builder(args)

