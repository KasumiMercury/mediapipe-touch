import {useRef, useState, useEffect, useCallback} from 'react'
import {FilesetResolver, GestureRecognizer, HandLandmarker} from "@mediapipe/tasks-vision";

function App() {
  const [cameraPermission, setCameraPermission] = useState<'idle' | 'requesting' | 'granted' | 'denied'>('idle')
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingMode, setProcessingMode] = useState<'hand' | 'gesture'>('hand')
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const requestCameraAccess = async () => {
    setCameraPermission('requesting')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: true})
      stream.getTracks().forEach(track => track.stop())
      setCameraPermission('granted')
    } catch (error) {
      console.error('Camera access denied:', error)
      setCameraPermission('denied')
    }
  }

  const initializeHandLandmarker = async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "/node_modules/@mediapipe/tasks-vision/wasm"
      )

      const handLandmarker = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        }
      )

      handLandmarkerRef.current = handLandmarker
      return handLandmarker
    } catch (error) {
      console.error('Failed to initialize HandLandmarker:', error)
      return null
    }
  }

  const initializeGestureRecognizer = async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "/node_modules/@mediapipe/tasks-vision/wasm"
      )

      const gestureRecognizer = await GestureRecognizer.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        }
      )

      gestureRecognizerRef.current = gestureRecognizer
      return gestureRecognizer
    } catch (error) {
      console.error('Failed to initialize Gesture Recognizer:', error)
      return null
    }
  }

  const startVideoStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      return stream;
    } catch (error) {
      console.error('Failed to start video stream:', error)
      return null
    }
  }

  const processHandLandmarker = () => {
    const video = videoRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !handLandmarker || video.readyState !== 4) {
      requestAnimationFrame(processHandLandmarker);
      return;
    }

    try {
      const startTimeMs = performance.now();
      const result = handLandmarker.detectForVideo(video, startTimeMs);

      if (result && result.landmarks.length > 0) {
        console.log(`Detected ${result.landmarks.length} hands:`, result.landmarks);
        
        result.landmarks.forEach((landmarks, handIndex) => {
          console.log(`Hand ${handIndex}:`, {
            thumbTip: landmarks[4],
            indexTip: landmarks[8],
            middleTip: landmarks[12],
            ringTip: landmarks[16],
            pinkyTip: landmarks[20]
          });
        });
      }
    } catch (error) {
      console.error('Error processing video frame:', error);
    }

    requestAnimationFrame(processHandLandmarker);
  }

  const processGestureRecognizer = () => {
    const video = videoRef.current;
    const gestureRecognizer = gestureRecognizerRef.current;

    if (!video || !gestureRecognizer || video.readyState !== 4) {
      requestAnimationFrame(processGestureRecognizer);
      return;
    }

    try {
      const startTimeMs = performance.now();
      const result = gestureRecognizer.recognizeForVideo(video, startTimeMs);
        if (result && result.gestures.length > 0) {
            console.log(`Detected ${result.gestures.length} gestures:`, result.gestures);

            result.gestures.forEach((gesture, index) => {
            console.log(`Gesture ${index}:`, gesture);
            });
        }
    } catch (error) {
      console.error('Error processing video frame:', error);
    }

    requestAnimationFrame(processGestureRecognizer);
  }

  const startProcessing = async () => {
    setIsProcessing(true);
    
    if (processingMode === 'hand') {
      const handLandmarker = await initializeHandLandmarker();
      if (!handLandmarker) {
        setIsProcessing(false);
        return;
      }

      const stream = await startVideoStream();
      if (!stream) {
        setIsProcessing(false);
        return;
      }

      processHandLandmarker();
    } else {
      const gestureRecognizer = await initializeGestureRecognizer();
      if (!gestureRecognizer) {
        setIsProcessing(false);
        return;
      }

      const stream = await startVideoStream();
      if (!stream) {
        setIsProcessing(false);
        return;
      }

      processGestureRecognizer();
    }
  }

  const stopProcessing = useCallback(() => {
    setIsProcessing(false);
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [])

  useEffect(() => {
    return () => {
      stopProcessing();
    };
  }, [stopProcessing]);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-white rounded-lg shadow-lg p-6">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            MediaPipe Hand Tracking & Gesture Recognition
          </h1>
        </div>

        <div className="space-y-4">
          {cameraPermission === 'idle' && (
            <button
              onClick={requestCameraAccess}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200"
            >
              カメラアクセスを許可
            </button>
          )}

          {cameraPermission === 'requesting' && (
            <div className="w-full bg-gray-100 text-gray-600 font-medium py-3 px-4 rounded-lg flex items-center justify-center gap-2">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-gray-300 border-t-blue-600"></div>
              カメラアクセスを要求中...
            </div>
          )}

          {cameraPermission === 'granted' && (
            <div className="space-y-4">
              <div className="w-full bg-green-100 text-green-800 font-medium py-3 px-4 rounded-lg text-center">
                カメラアクセス許可済み
              </div>
              
              <div className="flex gap-2 justify-center mb-4">
                <button
                  onClick={() => setProcessingMode('hand')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                    processingMode === 'hand' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  手のランドマーク
                </button>
                <button
                  onClick={() => setProcessingMode('gesture')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                    processingMode === 'gesture' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  ジェスチャ認識
                </button>
              </div>
              
              <div className="flex gap-4 justify-center">
                {!isProcessing ? (
                  <button
                    onClick={startProcessing}
                    className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200"
                  >
                    {processingMode === 'hand' ? '手の検出を開始' : 'ジェスチャ認識を開始'}
                  </button>
                ) : (
                  <button
                    onClick={stopProcessing}
                    className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200"
                  >
                    検出を停止
                  </button>
                )}
              </div>

              {isProcessing && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-blue-800 font-medium mb-2">
                    {processingMode === 'hand' ? '手のランドマーク検出中...' : 'ジェスチャ認識中...'}
                  </div>
                </div>
              )}
            </div>
          )}

          {cameraPermission === 'denied' && (
            <div className="space-y-3">
              <div className="w-full bg-red-100 text-red-800 font-medium py-3 px-4 rounded-lg text-center">
                カメラアクセスが拒否されました
              </div>
              <button
                onClick={requestCameraAccess}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
              >
                再試行
              </button>
            </div>
          )}
        </div>

        <video
          ref={videoRef}
          className="mt-6 w-full max-w-md mx-auto rounded-lg"
          style={{ display: isProcessing ? 'block' : 'none' }}
          autoPlay
          muted
          playsInline
        />
      </div>
    </div>
  )
}

export default App
