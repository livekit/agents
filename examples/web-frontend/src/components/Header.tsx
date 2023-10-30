import Link from "next/link";

export function Header() {
  return (
    <div className="flex w-full h-full bg-black items-center p-2 bg-gray-900">
      <Link href="/">LiveKit Agents Examples</Link>
    </div>
  );
}
