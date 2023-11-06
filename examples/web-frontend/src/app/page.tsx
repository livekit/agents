import Link from "next/link";

export default function Home() {
  return (
    <div>
      <div className="flex flex-col">
        <Link href={"/vad"}>VAD</Link>
        <Link href={"/stt"}>Speech-to-text</Link>
        <Link href={"/kitt"}>KITT (Talking AI)</Link>
        <Link href={"/tts"}>Text-to-speech</Link>
      </div>
    </div>
  );
}
