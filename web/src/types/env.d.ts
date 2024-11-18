declare namespace NodeJS {
  interface ProcessEnv {
    NEXT_PUBLIC_SITE_URL: string;
    NEXT_PUBLIC_VERCEL_URL: string;
    NEXT_PUBLIC_API_URL: string;

    NEXT_PUBLIC_MILVUS_ADDRESS: string;
  }
}
