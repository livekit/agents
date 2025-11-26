'use client';

import { useState } from 'react';
import { VideoRoom } from '@/components/VideoRoom';
import { DocumentManager } from '@/components/DocumentManager';
import { Analytics } from '@/components/Analytics';
import { FileText, Video, BarChart3 } from 'lucide-react';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'video' | 'documents' | 'analytics'>('video');
  const [roomName, setRoomName] = useState('');
  const [isInRoom, setIsInRoom] = useState(false);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <Video className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  RAG Video Platform
                </h1>
                <p className="text-sm text-gray-500">
                  AI Agent with Persistent Memory
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                Online
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="bg-white rounded-lg shadow-sm p-1 inline-flex space-x-1">
          <button
            onClick={() => setActiveTab('video')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'video'
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Video className="h-4 w-4" />
            <span>Video Session</span>
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'documents'
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <FileText className="h-4 w-4" />
            <span>Documents</span>
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'analytics'
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <BarChart3 className="h-4 w-4" />
            <span>Analytics</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 pb-12">
        {activeTab === 'video' && (
          <div className="space-y-6">
            {!isInRoom ? (
              <div className="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Start Video Session
                </h2>
                <p className="text-gray-600 mb-6">
                  Connect with an AI agent powered by RAG memory. The agent can access
                  uploaded documents and remember previous conversations.
                </p>
                <div className="space-y-4">
                  <div>
                    <label
                      htmlFor="roomName"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Session Name
                    </label>
                    <input
                      type="text"
                      id="roomName"
                      value={roomName}
                      onChange={(e) => setRoomName(e.target.value)}
                      placeholder="Enter a session name (e.g., your username)"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={() => {
                      if (roomName.trim()) {
                        setIsInRoom(true);
                      }
                    }}
                    disabled={!roomName.trim()}
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                  >
                    Start Video Session
                  </button>
                </div>
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold text-blue-900 mb-2">Features:</h3>
                  <ul className="space-y-1 text-sm text-blue-800">
                    <li>• Real-time video and voice interaction</li>
                    <li>• RAG-powered knowledge retrieval</li>
                    <li>• Persistent conversation memory</li>
                    <li>• Multi-modal AI understanding</li>
                  </ul>
                </div>
              </div>
            ) : (
              <VideoRoom
                roomName={roomName}
                onDisconnect={() => setIsInRoom(false)}
              />
            )}
          </div>
        )}

        {activeTab === 'documents' && <DocumentManager />}
        {activeTab === 'analytics' && <Analytics />}
      </div>
    </main>
  );
}
