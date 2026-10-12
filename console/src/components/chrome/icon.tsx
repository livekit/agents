import {
  Activity,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Copy,
  GitFork,
  Inbox,
  ListChecks,
  LoaderCircle,
  MessageSquare,
  MessagesSquare,
  Mic,
  MicOff,
  PanelLeftClose,
  Phone,
  PhoneOff,
  Plus,
  Radio,
  RotateCcw,
  ScrollText,
  SendHorizontal,
  SlidersHorizontal,
  Wrench,
  type LucideIcon,
} from "lucide-react";

/**
 * lucide at stroke 1.5, which at 16px is one CSS pixel: the same hairline
 * the panel edges are drawn with. Blender's own icons are filled, and the
 * weight of a solid shape says nothing about how heavy an outline should
 * be; heavier strokes read as bold next to 11px text.
 */
const ICONS = {
  conversations: MessagesSquare,
  conversation: MessageSquare,
  lanes: GitFork,
  tasks: ListChecks,
  journal: ScrollText,
  inbox: Inbox,
  stream: Radio,
  info: SlidersHorizontal,
  activity: Activity,
  replay: RotateCcw,
  send: SendHorizontal,
  plus: Plus,
  panelClose: PanelLeftClose,
  copy: Copy,
  check: Check,
  loader: LoaderCircle,
  tool: Wrench,
  mic: Mic,
  micOff: MicOff,
  phone: Phone,
  phoneOff: PhoneOff,
  voice: AudioLines,
  disclosureDown: ChevronDown,
  disclosureRight: ChevronRight,
  disclosureUp: ChevronUp,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;

export function Icon({ name, size = 16, className }: { name: IconName; size?: number; className?: string }) {
  const Glyph = ICONS[name];
  return <Glyph size={size} strokeWidth={1.5} className={className} aria-hidden="true" focusable="false" />;
}
