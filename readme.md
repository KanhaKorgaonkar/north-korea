# 🇰🇵 North Korea

> *"The People's Democratic Watermark Removal Software"*

Nobody's ever named a piece of software after a country. Until now.

---

## 🎬 About

**North Korea** is an AI-powered watermark remover specifically designed to remove those annoying moving Sora watermarks from your AI-generated videos. North Korea was built by Kanha Korgaonkar with Claude Sonnet 4.5 and won the mini hackathon at the Claude Builder's Club at the UW. 

### Origin Story
The other AI company released a video generation model that's blowing up the internet right now. Unfortunately, their API costs $0.10/s! That's $1 for just a 10 second clip!! While you can generate as many videos as you want for free on the Sora app, they have annoying moving watermarks that are moving all over the place so you can't even crop them out. When you post those videos to other platforms like TikTok or Instagram Reels, they're likely throttling them because of the watermark. <br>

That's why I built a tool to get rid of them. North Korea scans video frames for the Sora logo, learns its common positions, then inpaints/blur-fills those regions (while preserving audio) to spit out a watermark-free clip.
<br>
Special thanks to the **Claude Builders Club at UW** for hosting the event and the cool Claude "thinking cap" hat that I'm gonna wear everywhere.

---

## In action

<table>
<tr>
<td width="50%">
Before 😢
<br><br>
<video src="input_video.mp4" controls>
Your browser does not support the video tag.
</video>
<br>
Oppressive watermarks everywhere
</td>
<td width="50%">
After 🎉
<br><br>
<video src="output_cleaned.mp4" controls>
Your browser does not support the video tag.
</video>
<br>
Freedom achieved
</td>
</tr>
</table>

> Note: GitHub doesn't always render embedded videos. If you can't see them above, check the /examples folder directly!

## ✨ Features

- FAST Detection: Optimized watermark scanning so your GPU doesn’t explode.
- HYBRID Mode: Learns watermark positions, then keeps checking in case they start wandering like Kim Jong-un on a diplomatic tour.
- Adaptive Removal: Choose between inpainting, blur, or median color camouflage for obliterating logos.
- Audio Preservation: Thanks to ffmpeg, your beautiful voice-over propaganda survives intact.
- Progress Bars: So you can feel like a hacker while your video processes.

---

## 🚀 Installation

```bash
# Clone the repository (or just download the file, we're not picky)
git clone https://github.com/KanhaKorgaonkar/north-korea.git
cd north-korea

# Install dependencies
pip install opencv-python numpy

# Install ffmpeg (for audio preservation)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/ and add to PATH
```

---

## 📖 Usage

### Optional Customization and Steps to Run

1. **Prepare your watermark template**: This repo contains the Sora watermark as a reference by default but you can use this tool to get rid of other watermarks too. Screenshot a clear png image of the watermark and save it as `watermark_reference.png`

2. **Configure the script**: Also optional, but you can edit the configuration section at the bottom of the file:

```python
WATERMARK_TEMPLATE = 'watermark_reference.png'  # Contains a default Sora Video as a watermark 
INPUT_VIDEO = 'input_video.mp4'                 # Video to clean
OUTPUT_VIDEO = 'output_cleaned.mp4'             # Output file

CONFIDENCE_THRESHOLD = 0.65   # Detection sensitivity (0.55-0.75)
REMOVAL_METHOD = 'inpaint'    # 'inpaint', 'blur', or 'median'
CHECK_INTERVAL = 60           # Check for new positions every N frames
```

3. **Run it**:

```bash
python main.py
```

4. **Wait for freedom**: The script will analyze, process, and liberate your video from ugly watermarks.

---

## 🎮 Commands

Since this is a script and not a CLI tool (yet), you configure everything in the code (but you don't need to, the defaults work fine for most cases.):

| Configuration | Options | Description |
|--------------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.55` - `0.75` | Lower = more sensitive (catches more but may have false positives) |
| `REMOVAL_METHOD` | `'inpaint'` | Best quality, uses AI inpainting |
| | `'blur'` | Fast gaussian blur |
| | `'median'` | Fills with surrounding median color |
| `CHECK_INTERVAL` | Any integer | Frames between position checks (60 = ~2 seconds @ 30fps) |

---

## ❓ FAQ

**Q: Why "North Korea"?**  
A: Because I demoed this tool with an example Sora cameo with me in North Korea. Nobody's ever named a software after a country, and when I couldn't think of any other names for it before this launch, somebody suggested I just call it that. And I did, because it's funny. 

**Q: Is it legal to remove watermarks like this?**  
A: Relax, comrade. It’s just pixels. If you're worried about this being legal, you're probably a really boring and unhappy person. Touch grass. 

**Q: Does it work on other watermarks?**  
A: Absolutely! Just provide a template image of any watermark. It was designed for Sora's moving watermarks, but it'll work on any video watermark, like TikTok's. 

**Q: My watermark isn't being detected!**  
A: Try lowering `CONFIDENCE_THRESHOLD` to 0.60 or 0.55. Make sure your `watermark_reference.png` is a clear screenshot of the actual watermark.

**Q: Can I process multiple videos?**  
A: Currently one at a time, but you can easily loop it or modify the script. Pull requests welcome, comrade!

**Q: Did you really win the hackathon with this?**  
A: Yes! Popular vote via applause. Democracy works!

**Q: Can I get a Claude thinking hat too?**  
A: Attend Claude Builders Club events! They're really cool and they're spreading across universities.

---

## 🛠️ Technical Details

### How It Works

1. **Learning Phase**: Samples 50 frames throughout your video to learn common watermark positions
2. **Clustering**: Groups similar positions to identify unique locations
3. **Hybrid Processing**:
   - Uses learned positions for fast removal (no detection needed)
   - Periodically checks for new positions (catches moving watermarks)
   - Adapts in real-time if watermark moves to new locations
4. **Removal**: Uses OpenCV inpainting or other methods to seamlessly remove watermarks
5. **Audio Preservation**: Extracts, processes video, then merges audio back (requires ffmpeg)

### Performance

- **Speed**: 30-60 fps on modern hardware
- **Accuracy**: 90%+ detection rate with proper threshold tuning
- **Quality**: Inpainting method produces near-invisible removal

---

## 🤝 Contributing

Found a bug? Have a feature idea? Want to make North Korea even more glorious?

1. Fork the repo
2. Make your changes
3. Submit a pull request

All contributions welcome, from code to documentation to memes.

---

## 📜 License

MIT License - Do whatever you want with this code. Build upon it. Sell it. Name your firstborn after it. I don't care.

---

## 🙏 Acknowledgments

- **Claude Sonnet 4**: For being the AI that helped build this
- **UW Claude Builders Club**: For hosting the hackathon and the cool thinking hat
- **The People**: For voting via applause.
- **Dylan Pak**: For encouraging me to keep building this and demo it during the short hackathon. 
- **North Korea**: For obvious reasons.

---

## 📞 Support

Having issues? Questions? 

- Open an issue on GitHub
- Find me wearing my Claude thinking cap around campus

---

## 🎯 Roadmap

- [ ] CLI interface for easier configuration
- [ ] Batch processing for multiple videos
- [ ] Streamlit GUI for the non-technical masses
- [ ] Support for more watermark types
- [ ] World domination (maybe)

---

**Remember**: This tool is was built within 10 mins at a micro hackathon to win a hat. But it actually works really well. Enjoy your watermark-free content! 🎉

--
Built with love