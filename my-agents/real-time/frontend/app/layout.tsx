import { Public_Sans } from 'next/font/google';
import localFont from 'next/font/local';
import { headers } from 'next/headers';
import { GithubLogoIcon } from '@phosphor-icons/react/dist/ssr';
import { ThemeProvider } from '@/components/app/theme-provider';
import { ThemeToggle } from '@/components/app/theme-toggle';
import { cn } from '@/lib/shadcn/utils';
import { getAppConfig, getStyles } from '@/lib/utils';
import '@/styles/globals.css';

const publicSans = Public_Sans({
  variable: '--font-public-sans',
  subsets: ['latin'],
});

const commitMono = localFont({
  display: 'swap',
  variable: '--font-commit-mono',
  src: [
    {
      path: '../fonts/CommitMono-400-Regular.otf',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../fonts/CommitMono-700-Regular.otf',
      weight: '700',
      style: 'normal',
    },
    {
      path: '../fonts/CommitMono-400-Italic.otf',
      weight: '400',
      style: 'italic',
    },
    {
      path: '../fonts/CommitMono-700-Italic.otf',
      weight: '700',
      style: 'italic',
    },
  ],
});

interface RootLayoutProps {
  children: React.ReactNode;
}

export default async function RootLayout({ children }: RootLayoutProps) {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);
  const styles = getStyles(appConfig);
  const { pageTitle, pageDescription, companyName, logo, logoDark } = appConfig;

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        publicSans.variable,
        commitMono.variable,
        'scroll-smooth font-sans antialiased'
      )}
    >
      <head>
        {styles && <style suppressHydrationWarning dangerouslySetInnerHTML={{ __html: styles }} />}
        <title>{pageTitle}</title>
        <meta name="description" content={pageDescription} />
      </head>
      <body className="overflow-x-hidden">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <header
            className="border-border/60 bg-background/90 fixed top-0 left-0 flex w-full items-center justify-between border-b px-3 backdrop-blur-sm md:px-4"
            style={{
              height: 'calc(var(--app-top-strip-height) + env(safe-area-inset-top))',
              paddingTop: 'env(safe-area-inset-top)',
              zIndex: 'var(--app-z-header)',
            }}
          >
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://livekit.io"
              className="scale-100 transition-transform duration-300 hover:scale-110"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={logo}
                alt={`${companyName} Logo`}
                className="block size-14 sm:size-26 dark:hidden"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={logoDark ?? logo}
                alt={`${companyName} Logo`}
                className="hidden size-14 sm:size-26 dark:block"
              />
            </a>
            <div className="text-foreground flex items-center gap-3 text-right font-mono text-[11px] font-bold tracking-wider uppercase md:text-xs">
              <span>
                <span className="hidden sm:inline">Built with </span>
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href="https://docs.livekit.io/agents"
                  className="underline underline-offset-4"
                >
                  <span className="sm:hidden">LiveKit docs</span>
                  <span className="hidden sm:inline">LiveKit Agents</span>
                </a>
              </span>
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="https://github.com/livekit-examples/python-agents-examples/tree/main/complex-agents/avatars/anam"
                aria-label="Anam GitHub repository"
                className="text-foreground/80 hover:text-foreground transition-colors"
              >
                <GithubLogoIcon size={18} weight="fill" />
              </a>
            </div>
          </header>

          {children}
          <div className="group fixed bottom-0 left-1/2 z-50 mb-6 -translate-x-1/2 md:mb-2">
            <ThemeToggle className="translate-y-20 transition-transform delay-150 duration-300 group-hover:translate-y-0" />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
