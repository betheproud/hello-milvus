import { MilvusClient } from "@zilliz/milvus2-sdk-node";

// Milvus 클라이언트 인스턴스 생성
export const milvusClient = new MilvusClient({
  address: process.env.NEXT_PUBLIC_MILVUS_ADDRESS,
});
