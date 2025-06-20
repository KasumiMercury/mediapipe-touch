import {useRef, useState} from 'react'
import {FilesetResolver, HandLandmarker} from "@mediapipe/tasks-vision";

function App() {
  const [cameraPermission, setCameraPermission] = useState<'idle' | 'requesting' | 'granted' | 'denied'>('idle')

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

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const prepareVideoStreamWithVideoElements = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: true,
    });

    // video要素がまだなければ動的に作成
    let video = videoRef.current;
    if (!video) {
      console.log("video create")
      video = document.createElement('video');
      video.style.display = 'none'; // 非表示
      video.setAttribute('playsInline', '');
      video.setAttribute('muted', '');
      video.setAttribute('autoPlay', '');
      document.body.appendChild(video);
      videoRef.current = video;
    }
    video.srcObject = stream;
    video.onloadeddata = () => {
      process();
    };

    console.log(videoRef)
  };

  const process = async () => {

    const vision = await FilesetResolver.forVisionTasks(
        "node_modules/@mediapipe/tasks-vision/wasm"
    )

    const handLandmarker = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            // https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=ja#models
            // modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            modelAssetPath: "../hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        }
    )

    const debugLoop = () => {
      console.log("Running video stream...")

      const video = videoRef.current;
      if (!video) {
        console.log("Video element not found")
        return;
      }

      const result = handLandmarker.detectForVideo(video, performance.now());

      if (!result) {
        console.log("No result")
        return
      }

      if (result.landmarks.length > 0) {
        console.log("Landmarks:", result.landmarks)
      }

      requestAnimationFrame(() => {
        debugLoop();
      })
    }

    debugLoop()
  };

  return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="mb-6">
              <h1 className="text-2xl font-bold text-gray-900 mb-2">
                Camera Access
              </h1>
              <p className="text-gray-600">
                Please allow camera access to use this feature. Click the button below to request access.
              </p>
            </div>

            <div className="mb-6">
              <div className="w-fit mx-auto">
                {cameraPermission === 'idle' && (
                    <button
                        onClick={requestCameraAccess}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                    >
                      カメラを開始
                    </button>
                )}

                {cameraPermission === 'requesting' && (
                    <div
                        className="w-full bg-gray-100 text-gray-600 font-medium py-3 px-4 rounded-lg flex items-center justify-center gap-2">
                      <div
                          className="animate-spin rounded-full h-5 w-5 border-2 border-gray-300 border-t-blue-600"></div>
                      カメラアクセスを要求中...
                    </div>
                )}

                {cameraPermission === 'granted' && (
                    <div
                        className="w-full bg-green-100 text-green-800 font-medium py-3 px-4 rounded-lg flex items-center justify-center gap-2">
                      カメラアクセス許可済み
                    </div>
                )}

                {cameraPermission === 'denied' && (
                    <div className="space-y-3">
                      <div
                          className="w-full bg-red-100 text-red-800 font-medium py-3 px-4 rounded-lg flex items-center justify-center gap-2">
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
            </div>
          </div>
        </div>
      </div>
  )
}

export default App
