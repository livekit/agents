import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div>
      <div className="flex flex-col">
        <Link href={"/vad"}>VAD</Link>
        <Link href={"/stt"}>Speech-to-text</Link>
      </div>
    </div>
  );
}
