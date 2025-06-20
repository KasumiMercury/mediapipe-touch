import {useState} from 'react'

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
