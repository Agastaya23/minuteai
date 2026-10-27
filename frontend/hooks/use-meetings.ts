"use client";

import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { IN_FLIGHT } from "@/lib/format";

const POLL_MS = 3000;

/** Meeting list; polls every 3 s while any meeting is still processing. */
export function useMeetings(params: { series_id?: string } = {}) {
  return useQuery({
    queryKey: ["meetings", params],
    queryFn: () => api.listMeetings(params),
    refetchInterval: (q) => (q.state.data?.some((m) => IN_FLIGHT.has(m.status)) ? POLL_MS : false),
  });
}

export function useMeeting(id: string) {
  return useQuery({
    queryKey: ["meeting", id],
    queryFn: () => api.getMeeting(id),
    refetchInterval: (q) => (q.state.data && IN_FLIGHT.has(q.state.data.status) ? POLL_MS : false),
  });
}
