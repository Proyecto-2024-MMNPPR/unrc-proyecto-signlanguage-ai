export const SendButton = ({ onClick }: { onClick: () => void }) => (
  <button className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600" onClick={onClick}>
    Send Sequence
  </button>
);
