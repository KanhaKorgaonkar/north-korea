import cv2
import numpy as np
from collections import deque
import time
import subprocess
import os
import tempfile
import sys

class WatermarkRemover:
    def __init__(self, watermark_template_path, confidence_threshold=0.70):
        """Fast and precise watermark remover with adaptive detection"""
        self.template = cv2.imread(watermark_template_path, cv2.IMREAD_UNCHANGED)
        if self.template is None:
            raise ValueError("Could not load watermark template")
        
        if len(self.template.shape) == 3:
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            self.template_gray = self.template
            
        self.template_h, self.template_w = self.template_gray.shape[:2]
        self.confidence_threshold = confidence_threshold
        
        # Track positions over time
        self.known_positions = []
        self.position_history = deque(maxlen=100)
        self.last_detection_frame = 0
        
        print(f"✓ Template loaded: {self.template_w}x{self.template_h} pixels")
        print(f"✓ Confidence threshold: {confidence_threshold}")
    
    def detect_watermark_fast(self, frame):
        """FAST detection - optimized for speed"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        detections = []
        
        # Check multiple scales to catch size variations
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            scaled_w = int(self.template_w * scale)
            scaled_h = int(self.template_h * scale)
            
            if scaled_w > gray.shape[1] or scaled_h > gray.shape[0]:
                continue
            if scaled_w < 10 or scaled_h < 10:
                continue
            
            scaled_template = cv2.resize(self.template_gray, (scaled_w, scaled_h))
            
            # Template matching
            result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find all matches above threshold
            locations = np.where(result >= self.confidence_threshold)
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                detections.append((pt[0], pt[1], scaled_w, scaled_h, confidence))
        
        # Remove duplicates
        if len(detections) > 0:
            detections = self._remove_duplicates(detections)
        
        return detections
    
    def _remove_duplicates(self, detections):
        """Remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        for det in detections:
            # Check if this overlaps with any kept detection
            overlap = False
            for kept in keep:
                if self._boxes_overlap(det, kept):
                    overlap = True
                    break
            
            if not overlap:
                keep.append(det)
        
        return keep
    
    def _boxes_overlap(self, det1, det2, threshold=0.3):
        """Check if two detections overlap significantly"""
        x1, y1, w1, h1, _ = det1
        x2, y2, w2, h2, _ = det2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold
    
    def learn_positions(self, video_path, num_samples=50):
        """Learn watermark positions from video"""
        print("\n[Learning Phase] Analyzing video for watermark positions...")
        print("(This takes ~10-15 seconds but makes processing much faster)")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames throughout the video
        sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
        
        all_positions = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            detections = self.detect_watermark_fast(frame)
            all_positions.extend(detections)
            
            current = len([i for i in sample_indices if i <= idx])
            print(f"  Sampling: {current}/{num_samples} frames | Found {len(all_positions)} total detections", end='\r')
        
        cap.release()
        print()
        
        # Cluster positions to find common locations
        if len(all_positions) > 0:
            self.known_positions = self._cluster_positions(all_positions)
            print(f"✓ Found {len(self.known_positions)} unique watermark location(s):")
            for i, (x, y, w, h, conf) in enumerate(self.known_positions):
                print(f"  Location {i+1}: Position ({x}, {y}), Size {w}x{h}, Confidence {conf:.2%}")
        else:
            print("⚠ No watermarks detected in samples!")
            print("  Suggestions:")
            print("  - Lower confidence threshold (try 0.60 or 0.55)")
            print("  - Check that watermark_reference.png is correct")
            print("  - Make sure template matches watermark in video")
        
        return len(self.known_positions) > 0
    
    def _cluster_positions(self, positions):
        """Cluster similar positions together"""
        if not positions:
            return []
        
        # Group positions that are close to each other
        clusters = []
        
        for pos in positions:
            found_cluster = False
            
            for cluster in clusters:
                # Check if close to cluster center
                avg_x = np.mean([p[0] for p in cluster])
                avg_y = np.mean([p[1] for p in cluster])
                
                # Distance between position and cluster center
                dist = ((pos[0] - avg_x)**2 + (pos[1] - avg_y)**2)**0.5
                
                # If within 30 pixels, add to cluster
                if dist < 30:
                    cluster.append(pos)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append([pos])
        
        # Get representative position from each cluster
        representatives = []
        for cluster in clusters:
            # Sort by confidence and take top ones
            cluster = sorted(cluster, key=lambda x: x[4], reverse=True)
            
            # Average the top positions
            top_n = min(5, len(cluster))
            top = cluster[:top_n]
            
            avg_x = int(np.mean([p[0] for p in top]))
            avg_y = int(np.mean([p[1] for p in top]))
            avg_w = int(np.mean([p[2] for p in top]))
            avg_h = int(np.mean([p[3] for p in top]))
            max_conf = max([p[4] for p in top])
            
            representatives.append((avg_x, avg_y, avg_w, avg_h, max_conf))
        
        return representatives
    
    def update_positions(self, new_detections):
        """Update known positions with new detections"""
        for new_det in new_detections:
            # Check if this is a new position
            is_new = True
            
            for i, known in enumerate(self.known_positions):
                if self._boxes_overlap(new_det, known, threshold=0.5):
                    # Update existing position (weighted average)
                    x1, y1, w1, h1, c1 = known
                    x2, y2, w2, h2, c2 = new_det
                    
                    # Weighted by confidence
                    total_conf = c1 + c2
                    new_x = int((x1 * c1 + x2 * c2) / total_conf)
                    new_y = int((y1 * c1 + y2 * c2) / total_conf)
                    new_w = int((w1 * c1 + w2 * c2) / total_conf)
                    new_h = int((h1 * c1 + h2 * c2) / total_conf)
                    new_conf = max(c1, c2)
                    
                    self.known_positions[i] = (new_x, new_y, new_w, new_h, new_conf)
                    is_new = False
                    break
            
            if is_new:
                # Add new position
                self.known_positions.append(new_det)
    
    def remove_watermark_adaptive(self, frame, method='inpaint'):
        """Remove watermarks at known positions"""
        output = frame.copy()
        
        for x, y, w, h, conf in self.known_positions:
            # Add small padding for better coverage
            padding = 2
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            
            # Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            
            if method == 'inpaint':
                output = cv2.inpaint(output, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            elif method == 'blur':
                roi = output[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (11, 11), 0)
                output[y1:y2, x1:x2] = blurred
            elif method == 'median':
                expand = 8
                yy1 = max(0, y - expand)
                yy2 = min(frame.shape[0], y + h + expand)
                xx1 = max(0, x - expand)
                xx2 = min(frame.shape[1], x + w + expand)
                
                surrounding = frame[yy1:yy2, xx1:xx2]
                median_color = np.median(surrounding.reshape(-1, 3), axis=0).astype(np.uint8)
                output[y1:y2, x1:x2] = median_color
        
        return output
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _extract_audio(self, video_path, audio_path):
        """Extract audio from video"""
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', '-y', audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    
    def _merge_audio_video(self, video_path, audio_path, output_path):
        """Merge video with audio"""
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-y', output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    
    def process_video_hybrid(self, input_path, output_path, method='inpaint', 
                            check_interval=60):
        """
        HYBRID processing:
        - Learn positions first
        - Use learned positions for most frames (FAST)
        - Periodically check for new positions (catches moving watermarks)
        - Preserve audio
        
        Args:
            check_interval: Check for new watermarks every N frames (default: 60)
        """
        print(f"\n{'='*70}")
        print(f"HYBRID WATERMARK REMOVAL (Fast + Adaptive)")
        print(f"{'='*70}")
        
        # Step 0: Learn initial positions
        if not self.learn_positions(input_path, num_samples=50):
            print("\n✗ No watermarks detected. Check your settings.")
            return
        
        has_ffmpeg = self._check_ffmpeg()
        
        if not has_ffmpeg:
            print("\n⚠ WARNING: ffmpeg not found! Audio will be lost.\n")
            self._process_frames_hybrid(input_path, output_path, method, check_interval)
            return
        
        temp_dir = tempfile.gettempdir()
        temp_video = os.path.join(temp_dir, 'temp_video_no_audio.mp4')
        temp_audio = os.path.join(temp_dir, 'temp_audio.aac')
        
        try:
            # Extract audio
            print("\n[1/3] Extracting audio...")
            sys.stdout.flush()
            audio_extracted = self._extract_audio(input_path, temp_audio)
            if audio_extracted:
                print("✓ Audio extracted")
            else:
                print("⚠ No audio found")
            
            # Process video
            print(f"\n[2/3] Processing video (checking for new positions every {check_interval} frames)...")
            sys.stdout.flush()
            self._process_frames_hybrid(input_path, temp_video, method, check_interval)
            
            # Merge audio
            if audio_extracted and os.path.exists(temp_audio):
                print("\n[3/3] Merging audio...")
                sys.stdout.flush()
                merge_success = self._merge_audio_video(temp_video, temp_audio, output_path)
                if merge_success:
                    print("✓ Audio merged")
                else:
                    import shutil
                    shutil.copy(temp_video, output_path)
            else:
                import shutil
                shutil.copy(temp_video, output_path)
        
        finally:
            for temp_file in [temp_video, temp_audio]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        print(f"\n{'='*70}")
        print("✓ COMPLETE!")
        print(f"{'='*70}\n")
    
    def _process_frames_hybrid(self, input_path, output_path, method, check_interval):
        """Process frames with periodic detection checks"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f} seconds")
        print(f"  Method: {method}")
        print(f"  Mode: HYBRID (fast removal + periodic detection)")
        sys.stdout.flush()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_checks = 0
        new_positions_found = 0
        start_time = time.time()
        last_update_time = start_time
        
        print(f"\n{'='*70}")
        print("Processing...")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Periodically check for new watermark positions
                if frame_count % check_interval == 0:
                    new_detections = self.detect_watermark_fast(frame)
                    detection_checks += 1
                    
                    if new_detections:
                        old_count = len(self.known_positions)
                        self.update_positions(new_detections)
                        if len(self.known_positions) > old_count:
                            new_positions_found += len(self.known_positions) - old_count
                
                # Remove watermarks at known positions (FAST)
                cleaned_frame = self.remove_watermark_adaptive(frame, method)
                
                out.write(cleaned_frame)
                frame_count += 1
                
                # Update progress
                current_time = time.time()
                if frame_count % 10 == 0 or frame_count == total_frames or (current_time - last_update_time) > 0.5:
                    progress = (frame_count / total_frames) * 100
                    elapsed = current_time - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                    
                    bar_length = 30
                    filled = int(bar_length * frame_count / total_frames)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(f"\r[{bar}] {progress:5.1f}% | "
                          f"{frame_count}/{total_frames} | "
                          f"{fps_actual:5.1f} fps | "
                          f"ETA: {int(eta//60)}:{int(eta%60):02d} | "
                          f"Positions: {len(self.known_positions)}", 
                          end='', flush=True)
                    
                    last_update_time = current_time
        
        finally:
            print()
            cap.release()
            out.release()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ Processing complete!")
        print(f"  Frames: {frame_count}")
        print(f"  Detection checks: {detection_checks}")
        print(f"  New positions found: {new_positions_found}")
        print(f"  Final position count: {len(self.known_positions)}")
        print(f"  Time: {int(elapsed_time//60)}m {int(elapsed_time%60)}s")
        print(f"  Speed: {frame_count/elapsed_time:.1f} fps")
        print(f"{'='*70}")
        sys.stdout.flush()


# Main execution
if __name__ == "__main__":
    
    # ============ CONFIGURATION ============
    WATERMARK_TEMPLATE = 'watermark_reference.png'
    INPUT_VIDEO = 'input_video.mp4'
    OUTPUT_VIDEO = 'output_cleaned.mp4'
    
    CONFIDENCE_THRESHOLD = 0.65   # Lower = catch more (0.55-0.75)
    REMOVAL_METHOD = 'inpaint'    # 'inpaint', 'blur', or 'median'
    CHECK_INTERVAL = 60           # Check for new positions every N frames
    # ========================================
    
    try:
        if not os.path.exists(WATERMARK_TEMPLATE):
            print(f"ERROR: Template not found: {WATERMARK_TEMPLATE}")
            sys.exit(1)
        
        if not os.path.exists(INPUT_VIDEO):
            print(f"ERROR: Video not found: {INPUT_VIDEO}")
            sys.exit(1)
        
        print("Initializing HYBRID watermark remover...")
        print("(Combines speed of learned positions with adaptive detection)")
        sys.stdout.flush()
        
        remover = WatermarkRemover(
            watermark_template_path=WATERMARK_TEMPLATE,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Hybrid: Learn + periodic checks for moving watermarks
        remover.process_video_hybrid(
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
            method=REMOVAL_METHOD,
            check_interval=CHECK_INTERVAL  # Check every 60 frames (~2 seconds at 30fps)
        )
        
        print("\n✓ SUCCESS!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)