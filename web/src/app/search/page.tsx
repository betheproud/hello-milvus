"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface SearchResult {
  comment: string;
  rating: number;
  product_id: number;
  similarity: number;
}

export default function SearchPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const searchReviews = async () => {
    if (!searchTerm.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: searchTerm,
          limit: 100,
        }),
      });

      if (!response.ok) {
        throw new Error("검색 중 오류가 발생했습니다.");
      }

      const results = await response.json();
      setSearchResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "오류가 발생했습니다.");
      console.error("Search error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full h-full flex flex-col items-center space-y-2">
      <Label className="text-xl font-black">Vector DB Search by Review</Label>
      <div className="w-full h-full p-4 flex flex-col">
        {/* 검색 폼 */}
        <div className="mb-4 flex gap-2">
          <Input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="리뷰 내용을 입력하세요"
            className="text-sm"
          />
          <Button
            onClick={searchReviews}
            disabled={isLoading || !searchTerm.trim()}
          >
            {isLoading ? "검색중..." : "검색"}
          </Button>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className="mb-4 p-4 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}

        {/* 검색 결과 */}
        <div className="flex-1">
          {searchResults.length > 0 ? (
            <div>
              <Label className="text-lg font-black">
                검색 결과: {searchResults.length}건
              </Label>
              <div className="space-y-3 mt-2">
                {searchResults.map((result, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <Label className="text-sm text-gray-500">
                        유사도: {(result.similarity * 100).toFixed(2)}%
                      </Label>
                    </CardHeader>
                    <CardContent className="flex flex-col space-y-1">
                      <Label className="text-sm">{result.comment}</Label>
                      <Label className="text-xs">
                        Product ID: {result.product_id}
                      </Label>
                      <div className="flex justify-between items-start">
                        <div className="flex items-center">
                          <span className="text-yellow-500">★</span>
                          <Label className="ml-1">
                            {result.rating.toFixed(1)}
                          </Label>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ) : (
            !isLoading &&
            searchTerm && (
              <div className="text-center text-gray-500">
                검색 결과가 없습니다.
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
