This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## CLI Commands

The following CLI commands are available for working with Megaton-roto:

| Command | Description |
|---------|-------------|
| `backlog` | View real-time logs from the main Megaton-roto service using journalctl |
| `frontlog` | View real-time logs from the frontend service using journalctl |
| `backend` | Navigate to the backend directory and start the backend server |

These commands are aliases that can be added to your shell configuration for convenience. They help with monitoring logs and starting services during development.

### Port Management

The command `sudo fuser -k 3000/tcp` is used to forcefully terminate any process that is currently using port 3000. This is particularly useful when:

1. You get an error that port 3000 is already in use when trying to start the development server
2. The previous Next.js development server didn't shut down properly
3. You need to quickly free up port 3000 for a new instance of the application

**Note:** Use this command with caution as it will kill any process using port 3000 without warning. Make sure you don't have any important processes running on this port before executing the command.

sudo fuser -k 3000/tcp

## Getting Started

First, run the development server:

cd ~/megaton-roto/frontend && npm run dev

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```
cd 
Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

sudo journalctl -u megaton-roto.service -f
