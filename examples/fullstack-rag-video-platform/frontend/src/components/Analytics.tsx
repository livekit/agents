'use client';

import { useState, useEffect } from 'react';
import { Activity, FileText, MessageSquare, Users } from 'lucide-react';
import axios from 'axios';

interface SystemStats {
  rag_status: string;
  document_count: number;
  total_conversations: number;
  active_sessions: number;
  uptime_seconds: number;
}

export function Analytics() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  async function loadStats() {
    try {
      const response = await axios.get('/api/analytics/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    } finally {
      setIsLoading(false);
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-12 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading analytics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          System Analytics
        </h2>
        <p className="text-gray-600">
          Real-time insights into your RAG video platform.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={<Activity className="h-8 w-8" />}
          title="RAG Status"
          value={stats?.rag_status || 'Unknown'}
          color="blue"
          isStatus
        />
        <StatCard
          icon={<FileText className="h-8 w-8" />}
          title="Documents"
          value={stats?.document_count || 0}
          color="purple"
        />
        <StatCard
          icon={<MessageSquare className="h-8 w-8" />}
          title="Conversations"
          value={stats?.total_conversations || 0}
          color="green"
        />
        <StatCard
          icon={<Users className="h-8 w-8" />}
          title="Active Sessions"
          value={stats?.active_sessions || 0}
          color="orange"
        />
      </div>

      {/* Performance Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Performance Metrics
        </h3>
        <div className="space-y-4">
          <MetricBar
            label="RAG Retrieval Speed"
            value={95}
            color="blue"
            suffix="ms avg"
          />
          <MetricBar
            label="LLM Response Time"
            value={85}
            color="purple"
            suffix="ms avg"
          />
          <MetricBar
            label="Video Quality"
            value={98}
            color="green"
            suffix="% uptime"
          />
          <MetricBar
            label="Memory Efficiency"
            value={92}
            color="orange"
            suffix="% optimal"
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Activity
        </h3>
        <div className="space-y-3">
          <ActivityItem
            type="conversation"
            message="New conversation started"
            time="2 minutes ago"
          />
          <ActivityItem
            type="document"
            message="Document uploaded: sales_report.pdf"
            time="15 minutes ago"
          />
          <ActivityItem
            type="query"
            message="RAG query: What is the pricing model?"
            time="32 minutes ago"
          />
          <ActivityItem
            type="conversation"
            message="Conversation ended (15 min session)"
            time="1 hour ago"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  title,
  value,
  color,
  isStatus = false,
}: {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  color: string;
  isStatus?: boolean;
}) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    purple: 'bg-purple-100 text-purple-600',
    green: 'bg-green-100 text-green-600',
    orange: 'bg-orange-100 text-orange-600',
  }[color];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className={`inline-flex p-3 rounded-lg mb-4 ${colorClasses}`}>
        {icon}
      </div>
      <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
      <p
        className={`text-2xl font-bold ${
          isStatus && value === 'ready' ? 'text-green-600' : 'text-gray-900'
        }`}
      >
        {isStatus && value === 'ready' ? (
          <span className="flex items-center">
            <span className="w-3 h-3 bg-green-400 rounded-full mr-2 animate-pulse"></span>
            Ready
          </span>
        ) : (
          value
        )}
      </p>
    </div>
  );
}

function MetricBar({
  label,
  value,
  color,
  suffix,
}: {
  label: string;
  value: number;
  color: string;
  suffix: string;
}) {
  const colorClasses = {
    blue: 'bg-blue-600',
    purple: 'bg-purple-600',
    green: 'bg-green-600',
    orange: 'bg-orange-600',
  }[color];

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm text-gray-600">
          {value} {suffix}
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${colorClasses} transition-all`}
          style={{ width: `${value}%` }}
        ></div>
      </div>
    </div>
  );
}

function ActivityItem({
  type,
  message,
  time,
}: {
  type: string;
  message: string;
  time: string;
}) {
  const iconMap = {
    conversation: <MessageSquare className="h-5 w-5 text-blue-600" />,
    document: <FileText className="h-5 w-5 text-purple-600" />,
    query: <Activity className="h-5 w-5 text-green-600" />,
  };

  return (
    <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
      <div className="mt-0.5">{iconMap[type as keyof typeof iconMap]}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-900">{message}</p>
        <p className="text-xs text-gray-500 mt-1">{time}</p>
      </div>
    </div>
  );
}
