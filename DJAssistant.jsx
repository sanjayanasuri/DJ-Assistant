import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { motion } from "framer-motion";
import axios from "axios";

export default function DJAssistant() {
  const [currentTrack, setCurrentTrack] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetchInitialRecommendation();
  }, []);

  const fetchInitialRecommendation = async () => {
    try {
      const response = await axios.get("http://localhost:5000/recommend");
      setCurrentTrack(response.data.currentTrack);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error("Error fetching initial recommendation:", error);
    }
  };

  const handleFeedback = async (trackId, feedback) => {
    try {
      await axios.post("http://localhost:5000/feedback", { trackId, feedback });
      fetchInitialRecommendation();
    } catch (error) {
      console.error("Error sending feedback:", error);
    }
  };

  const handleSearch = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/search?query=${searchQuery}`);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error("Error searching tracks:", error);
    }
  };

  return (
    <div className="flex flex-col items-center p-6">
      <h1 className="text-2xl font-bold mb-4">DJ Assistant</h1>
      {currentTrack && (
        <Card className="w-full max-w-md mb-6">
          <CardContent className="p-4">
            <h2 className="text-xl font-semibold">Now Playing</h2>
            <p className="text-lg">{currentTrack.title} - {currentTrack.artist}</p>
          </CardContent>
        </Card>
      )}
      <Input 
        className="mb-4" 
        placeholder="Search for a track..." 
        value={searchQuery} 
        onChange={(e) => setSearchQuery(e.target.value)}
      />
      <Button className="mb-6" onClick={handleSearch}>Search</Button>
      <div className="grid gap-4 w-full max-w-md">
        {recommendations.map((track) => (
          <motion.div key={track.id} whileHover={{ scale: 1.05 }}>
            <Card className="p-4">
              <CardContent>
                <p className="text-lg">{track.title} - {track.artist}</p>
                <div className="flex justify-between mt-2">
                  <Button onClick={() => handleFeedback(track.id, 1)}>👍</Button>
                  <Button onClick={() => handleFeedback(track.id, 0)}>👎</Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
