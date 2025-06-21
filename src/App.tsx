import {FilesetResolver, GestureRecognizer, HandLandmarker} from "@mediapipe/tasks-vision";
import {useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const [cameraPermission, setCameraPermission] = useState<'idle' | 'requesting' | 'granted' | 'denied'>('idle')
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingMode, setProcessingMode] = useState<'hand' | 'gesture'>('hand')
  const [fingerPositions, setFingerPositions] = useState<{ [handIndex: number]: { x: number; y: number } }>({})
  const [interactiveElements, setInteractiveElements] = useState<{id: string, isActive: boolean}[]>([
    {id: 'element1', isActive: false},
    {id: 'element2', isActive: false}
  ])
  const element1Ref = useRef<HTMLDivElement | null>(null)
  const element2Ref = useRef<HTMLDivElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const pointGesture = "Pointing_Up"
  const gestureThreshold = 0.1
  const indexFingerTip = 8
  const verticalOffset = 0
  const detectionScale = 0.8
  const handColors = ['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500']

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
          width: {ideal: 640},
          height: {ideal: 480},
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

  const convertToScreenPosition = (x: number, y: number) => {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const screenAspect = screenWidth / screenHeight;
    const videoAspect = 640 / 480;

    const scaleRatio = screenAspect / videoAspect;
    
    const videoScaledX = x;
    const videoScaledY = (y - 0.5) * scaleRatio + 0.5;

    const scaledX = (videoScaledX - 0.5) / detectionScale + 0.5;
    const scaledY = (videoScaledY - 0.5) / detectionScale + 0.5;

    if (scaledX < 0 || scaledX > 1 || scaledY < 0 || scaledY > 1) {
      return null;
    }

    const flippedX = 1 - scaledX;

    const adjustedX = flippedX * screenWidth;
    const adjustedY = scaledY * screenHeight + verticalOffset;

    return {x: adjustedX, y: adjustedY};
  }

  const checkCollision = useCallback((fingerPos: {x: number, y: number}, element: HTMLDivElement) => {
    const rect = element.getBoundingClientRect();
    return fingerPos.x >= rect.left && 
           fingerPos.x <= rect.right && 
           fingerPos.y >= rect.top && 
           fingerPos.y <= rect.bottom;
  }, [])

  useEffect(() => {
    if (Object.keys(fingerPositions).length === 0) {
      setInteractiveElements(prev => prev.map(el => ({...el, isActive: false})));
      return;
    }

    const elements = [element1Ref.current, element2Ref.current];
    
    setInteractiveElements(prev => prev.map((el, index) => {
      const element = elements[index];
      if (!element) return el;
      
      const isActive = Object.values(fingerPositions).some(pos => checkCollision(pos, element));
      return {...el, isActive};
    }));
  }, [fingerPositions, checkCollision])

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
        const newPositions: { [handIndex: number]: { x: number; y: number } } = {};
        result.landmarks.forEach((landmarks, handIndex) => {
          // console.log(`Detected Index Tip of ${handIndex}:`, landmarks[indexFingerTip]);
          const indexTip = landmarks[indexFingerTip];
          if (indexTip) {
            const screenPos = convertToScreenPosition(indexTip.x, indexTip.y);
            if (screenPos) {
              newPositions[handIndex] = screenPos;
            }
          }
        });
        setFingerPositions(newPositions);
      } else {
        setFingerPositions({});
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
        const newPositions: { [handIndex: number]: { x: number; y: number } } = {};
        result.gestures.forEach((gesture, handIndex) => {
          if (gesture[0].categoryName !== pointGesture) {
            return;
          }

          if (gesture[0].score < gestureThreshold) {
            return;
          }

          const indexTip = result.landmarks[handIndex][indexFingerTip];
          if (indexTip) {
            const screenPos = convertToScreenPosition(indexTip.x, indexTip.y);
            if (screenPos) {
              newPositions[handIndex] = screenPos;
            }
          }
        })

        setFingerPositions(newPositions);
      } else {
        setFingerPositions({});
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
    setFingerPositions({});

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
      <>
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
                  <div
                      className="w-full bg-gray-100 text-gray-600 font-medium py-3 px-4 rounded-lg flex items-center justify-center gap-2">
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
                style={{
                  display: isProcessing ? 'block' : 'none',
                  transform: 'scaleX(-1)'
                }}
                autoPlay
                muted
                playsInline
            />
          </div>

          {Object.entries(fingerPositions).map(([handIndex, position]) => (
            <div
              key={handIndex}
              className={`fixed w-6 h-6 ${handColors[parseInt(handIndex) % handColors.length]} rounded-full pointer-events-none z-50 border-2 border-white shadow-lg`}
              style={{
                left: position.x - 12,
                top: position.y - 12,
                transform: 'translate(0, 0)',
              }}
            />
          ))}

        <div className="fixed inset-0 pointer-events-none z-40">
          <div 
            ref={element1Ref}
            className={`absolute w-32 h-32 ${interactiveElements[0]?.isActive ? 'bg-red-500' : 'bg-blue-500'} opacity-70 rounded-lg transition-colors duration-200`}
            style={{
              left: '20%',
              top: '30%'
            }}
          />
          <div 
            ref={element2Ref}
            className={`absolute w-32 h-32 ${interactiveElements[1]?.isActive ? 'bg-red-500' : 'bg-green-500'} opacity-70 rounded-lg transition-colors duration-200`}
            style={{
              right: '20%',
              top: '50%'
            }}
          />
        </div>
        </div>
      </>
  )
}

export default App
