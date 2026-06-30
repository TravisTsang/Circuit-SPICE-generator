export default function Loading() {
  return (
    <div className="mx-auto grid min-h-[60dvh] w-full max-w-7xl place-items-center px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-xl rounded-lg border border-line/70 bg-surface/75 p-5">
        <div className="h-3 w-24 rounded bg-line/55" />
        <div className="mt-5 h-8 w-3/4 rounded bg-line/45" />
        <div className="mt-4 h-3 w-full rounded bg-line/35" />
        <div className="mt-2 h-3 w-5/6 rounded bg-line/35" />
      </div>
    </div>
  );
}
