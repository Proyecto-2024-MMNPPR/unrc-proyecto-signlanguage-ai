export const VideoAndCanvas = ({ videoRef, canvasRef }: { videoRef: any, canvasRef: any }) => (
  <div className='flex justify-center items-center gap-4 mt-2'>
    <div className="w-1/2 h-auto rounded-lg">
      <video ref={videoRef} className='w-full h-auto rounded-lg' style={{ objectFit: 'cover' }} />
    </div>
    <div className="w-1/2 h-auto rounded-lg">
      <canvas ref={canvasRef} width="640" height="480" className="w-full h-auto rounded-lg" style={{ objectFit: "cover" }} />
    </div>
  </div>
);
