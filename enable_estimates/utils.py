import itertools
from dataclasses import dataclass, fields
import pandas as pd
import datetime
from datetime import timedelta
import pytz
from typing import Optional


int_week_days = {
    0: "MONDAY",
    1: "TUESDAY",
    2: "WEDNESDAY",
    3: "THURSDAY",
    4: "FRIDAY",
    5: "SATURDAY",
    6: "SUNDAY",
}

str_week_days = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}

conversion_table = {
    "WEEKS": 604800,
    "DAYS": 86400,
    "HOURS": 3600,
    "MINUTES": 60,
    "SECONDS": 1,
}



def zip_with_next(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

@dataclass
class EventLogIDs:
    # General
    case: str = "case"  # Case ID
    activity: str = "activity"  # Activity label
    resource: str = "resource"  # Resource who performed this activity instance
    start_time: str = "start_time"  # Start time of the activity instance
    end_time: str = "end_time"  # End time of the activity instance
    # Start time estimator
    enabled_time: str = "enabled_time"  # Enablement time of the activity instance
    enabling_activity: str = "enabling_activity"  # Label of the activity instance enabling the current one
    enabling_resource: str = "enabling_resource"  # Label of the activity instance enabling the current one
    available_time: str = (
        "available_time"  # Last availability time of the resource who performed this activity instance
    )
    estimated_start_time: str = "estimated_start_time"  # Estimated start time of the activity instance
    batch_id: str = "batch_instance_id"  # ID of the batch instance this activity instance belongs to, if any
    batch_type: str = "batch_instance_type"  # Type of the batch instance this activity instance belongs to, if any

    @staticmethod
    def from_dict(config: dict) -> "EventLogIDs":
        return EventLogIDs(**config)

    def to_dict(self) -> dict:
        return {attr.name: getattr(self, attr.name) for attr in fields(self.__class__)}


class IntervalPoint:
    def __init__(self, date_time, week_day, index, to_start_dist, to_end_dist):
        self.date_time = date_time
        self.week_day = week_day
        self.index = index
        self.to_start_dist = to_start_dist
        self.to_end_dist = to_end_dist

    def in_same_interval(self, another_point):
        return self.week_day == another_point.week_day and self.index == another_point.index


@dataclass
class Interval:
    start: datetime.datetime
    end: datetime.datetime
    duration: float

    def __init__(self, start: datetime.datetime, end: datetime.datetime):
        self.start = start
        if end < start and end.hour == 0 and end.minute == 0:
            end.replace(hour=23, minute=59, second=59, microsecond=999)
        self.end = end
        self.duration = (end - start).total_seconds()

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.start == other.start and self.end == other.end
        else:
            return False

    def merge_interval(self, n_interval: "Interval"):
        self.start = min(n_interval.start, self.start)
        self.end = max(n_interval.end, self.end)
        self.duration = (self.end - self.start).total_seconds()

    def is_before(self, c_date):
        return self.end <= c_date

    def contains(self, c_date):
        return self.start < c_date < self.end

    def contains_inclusive(self, c_date):
        return self.start <= c_date <= self.end

    def is_after(self, c_date):
        return c_date <= self.start

    def intersection(self, interval):
        if interval is None:
            return None
        [first_i, second_i] = [self, interval] if self.start <= interval.start else [interval, self]
        if second_i.start < first_i.end:
            return Interval(max(first_i.start, second_i.start), min(first_i.end, second_i.end))
        return None
    

class CalendarIterator:
    def __init__(self, start_date: datetime, calendar_info):
        self.start_date = start_date

        self.calendar = calendar_info

        self.c_day = start_date.date().weekday()

        c_date = datetime.datetime.combine(calendar_info.default_date, start_date.time())
        c_interval = calendar_info.work_intervals[self.c_day][0]
        self.c_index = -1
        while c_interval.end < c_date and self.c_index < len(calendar_info.work_intervals[self.c_day]) - 1:
            self.c_index += 1
            c_interval = calendar_info.work_intervals[self.c_day][self.c_index]

        self.c_interval = Interval(
            self.start_date,
            self.start_date + timedelta(seconds=(c_interval.end - c_date).total_seconds()),
        )

    def next_working_interval(self):
        res_interval = self.c_interval
        day_intervals = self.calendar.work_intervals[self.c_day]
        p_duration = 0

        self.c_index += 1
        if self.c_index >= len(day_intervals):
            p_duration += 86400 - (day_intervals[self.c_index - 1].end - self.calendar.new_day).total_seconds()
            while True:
                self.c_day = (self.c_day + 1) % 7
                day_intervals = self.calendar.work_intervals[self.c_day]
                if len(day_intervals) > 0:
                    p_duration += (day_intervals[0].start - self.calendar.new_day).total_seconds()
                    break
                else:
                    p_duration += 86400
            self.c_index = 0
        elif self.c_index > 0:
            p_duration += (day_intervals[self.c_index].start - day_intervals[self.c_index - 1].end).total_seconds()
        self.c_interval = Interval(
            res_interval.end + timedelta(seconds=p_duration),
            res_interval.end + timedelta(seconds=p_duration + day_intervals[self.c_index].duration),
        )
        return res_interval

def to_seconds(value, from_unit):
    u_from = from_unit.upper()
    return value * conversion_table[u_from] if u_from in conversion_table else value

    
class RCalendar:  # AvailabilityCalendar
    def __init__(self, calendar_id):
        self.calendar_id = calendar_id
        self.default_date = None
        self.work_intervals = {}
        self.new_day = None
        self.cumulative_work_durations = {}
        self.work_rest_count = {}
        self.total_weekly_work = 0
        self.total_weekly_rest = to_seconds(1, "WEEKS")
        for i in range(0, 7):
            self.work_intervals[i] = []
            self.cumulative_work_durations[i] = []
            self.work_rest_count[i] = [0, to_seconds(1, "DAYS")]

    def to_dict(self):
        return {
            "id": self.calendar_id,
            "name": self.calendar_id,
            "time_periods": self.intervals_to_json(),
        }

    def intervals_to_json(self):
        items = []

        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                for interval in self.work_intervals[i]:
                    items.append(
                        {
                            "from": int_week_days[i],
                            "to": int_week_days[i],
                            "beginTime": str(interval.start.time()),
                            "endTime": str(interval.end.time()),
                        }
                    )

        return items

    @staticmethod
    def from_dict(calendar_dict):
        calendar = RCalendar(calendar_dict["id"])
        for time_period in calendar_dict["time_periods"]:
            calendar.add_calendar_item(
                time_period["from"],
                time_period["to"],
                time_period["beginTime"],
                time_period["endTime"],
            )
        return calendar

    def is_empty(self):
        # Return false (no empty) if any interval in a week day
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                return False
        # Return true (empty) if no interval found
        return True

    def is_working_datetime(self, date_time):
        c_day = date_time.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, date_time.time())
        i_index = 0
        for interval in self.work_intervals[c_day]:
            if interval.contains_inclusive(c_date):
                return True, IntervalPoint(
                    date_time,
                    i_index,
                    c_day,
                    (c_date - interval.start).total_seconds(),
                    (interval.end - c_date).total_seconds(),
                )
            i_index += 1
        return False, None

    def combine_calendar(self, new_calendar):
        for i in range(0, 7):
            if len(new_calendar.work_intervals[i]) > 0:
                for interval in new_calendar.work_intervals[i]:
                    self.add_calendar_item(
                        int_week_days[i],
                        int_week_days[i],
                        str(interval.start.time()),
                        str(interval.end.time()),
                    )

    def add_calendar_item(self, from_day: str, to_day: str, begin_time: str, end_time: str):
        if from_day.upper() in str_week_days and to_day.upper() in str_week_days:
            try:
                t_interval = Interval(
                    start=pd.Timestamp(begin_time).to_pydatetime(),
                    end=pd.Timestamp(end_time).to_pydatetime(),
                )
                if self.default_date is None:
                    self.default_date = t_interval.start.date()
                    self.new_day = datetime.datetime.combine(self.default_date, datetime.time())
                d_s = str_week_days[from_day]
                d_e = str_week_days[to_day]
                while True:
                    self._add_interval(d_s % 7, t_interval)
                    if d_s % 7 == d_e:
                        break
                    d_s += 1
            except ValueError:
                return

    def _add_interval(self, w_day, interval):
        i = 0
        for to_merge in self.work_intervals[w_day]:
            if to_merge.end < interval.start:
                i += 1
                continue
            if interval.end < to_merge.start:
                break
            merged_duration = to_merge.duration
            to_merge.merge_interval(interval)
            merged_duration = to_merge.duration - merged_duration
            i += 1
            while i < len(self.work_intervals[w_day]):
                next_i = self.work_intervals[w_day][i]
                if to_merge.end < next_i.start:
                    break
                if next_i.end <= to_merge.end:
                    merged_duration -= next_i.duration
                elif next_i.start <= to_merge.end:
                    merged_duration -= (to_merge.end - next_i.start).total_seconds()
                    to_merge.merge_interval(next_i)
                del self.work_intervals[w_day][i]
            if merged_duration > 0:
                self._update_calendar_durations(w_day, merged_duration)
            return
        self.work_intervals[w_day].insert(i, interval)
        self._update_calendar_durations(w_day, interval.duration)

    def compute_cumulative_durations(self):
        for w_day in self.work_intervals:
            cumulative = 0
            for interval in self.work_intervals[w_day]:
                cumulative += interval.duration
                self.cumulative_work_durations[w_day].append(cumulative)

    def _update_calendar_durations(self, w_day, duration):
        self.work_rest_count[w_day][0] += duration
        self.work_rest_count[w_day][1] -= duration
        self.total_weekly_work += duration
        self.total_weekly_rest -= duration

    def remove_idle_times(self, from_date, to_date, out_intervals: list):
        calendar_it = CalendarIterator(from_date, self)
        while True:
            c_interval = calendar_it.next_working_interval()
            if c_interval.end < to_date:
                out_intervals.append(c_interval)
            else:
                if c_interval.start <= to_date <= c_interval.end:
                    out_intervals.append(Interval(c_interval.start, to_date))
                break

    def find_idle_time(self, requested_date, duration):
        if duration == 0:
            return 0
        real_duration = 0
        pending_duration = duration
        if duration > self.total_weekly_work:
            real_duration += to_seconds(int(duration / self.total_weekly_work), "WEEKS")
            pending_duration %= self.total_weekly_work
        # Addressing the first day as an special case
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        worked_time, total_time = self._find_time_starting(pending_duration, c_day, c_date)
        if worked_time > total_time and worked_time - total_time < 0.001:
            total_time = worked_time
        pending_duration -= worked_time
        real_duration += total_time
        c_date = self.new_day
        while pending_duration > 0:
            c_day += 1
            r_d = c_day % 7
            if pending_duration > self.work_rest_count[r_d][0]:
                pending_duration -= self.work_rest_count[r_d][0]
                real_duration += 86400
            else:
                real_duration += self._find_time_completion(pending_duration, self.work_rest_count[r_d][0], r_d, c_date)
                break
        return real_duration

    def next_available_time(self, requested_date):
        """
        Validates whether the 'requested_date' is located in the arrival time calendar.
        Valid = complies with the provided time periods of the arrival calendar.
        If the 'requested_date' is valid, 0 is being returned (no waiting time).
        If not, the number of seconds we need to wait till the time the datetime will
        become eligible and comply with the time periods of the calendar.
        """
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        for interval in self.work_intervals[c_day]:
            if interval.end == c_day:
                continue
            if interval.is_after(c_date):
                return (interval.start - c_date).total_seconds()
            if interval.contains(c_date):
                return 0
        duration = 86400 - (c_date - self.new_day).total_seconds()
        for i in range(c_day + 1, c_day + 8):
            r_day = i % 7
            if self.work_rest_count[r_day][0] > 0:
                return duration + (self.work_intervals[r_day][0].start - self.new_day).total_seconds()
            duration += 86400
        return duration

    def find_working_time(self, start_date, end_date):
        pending_duration = (end_date - start_date).total_seconds()
        worked_hours = 0

        c_day = start_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, start_date.time())

        to_complete_day = 86400 - (c_date - self.new_day).total_seconds()
        available_work = self._calculate_available_duration(c_day, c_date)

        previous_date = c_date
        while pending_duration > to_complete_day:
            pending_duration -= to_complete_day
            worked_hours += available_work
            c_day = (c_day + 1) % 7
            available_work = self.work_rest_count[c_day][0]
            to_complete_day = 86400
            previous_date = self.new_day

        for interval in self.work_intervals[c_day]:
            if interval.is_before(previous_date):
                continue
            interval_duration = interval.duration
            if interval.contains(previous_date):
                interval_duration -= (previous_date - interval.start).total_seconds()
            else:
                pending_duration -= (interval.start - previous_date).total_seconds()
            if pending_duration >= interval_duration:
                worked_hours += interval_duration
            elif pending_duration > 0:
                worked_hours += pending_duration
            pending_duration -= interval_duration
            if pending_duration <= 0:
                break
            previous_date = interval.end
        return worked_hours

    def _find_time_starting(self, pending_duration, c_day, from_date):
        available_duration = self._calculate_available_duration(c_day, from_date)
        if available_duration <= pending_duration:
            return (
                available_duration,
                86400 - (from_date - self.new_day).total_seconds(),
            )
        else:
            return pending_duration, self._find_time_completion(pending_duration, available_duration, c_day, from_date)

    def _calculate_available_duration(self, c_day, from_date):
        i = -1
        passed_duration = 0
        for t_interval in self.work_intervals[c_day]:
            i += 1
            if t_interval.is_before(from_date):
                passed_duration += t_interval.duration
                continue
            if t_interval.is_after(from_date):
                break
            if t_interval.contains(from_date):
                passed_duration += (from_date - self.work_intervals[c_day][i].start).total_seconds()
                break

        return self.work_rest_count[c_day][0] - passed_duration

    def _find_time_completion(self, pending_duration, total_duration, c_day, from_datetime):
        i = len(self.work_intervals[c_day]) - 1
        while total_duration > pending_duration:
            total_duration -= self.work_intervals[c_day][i].duration
            i -= 1
        if total_duration < pending_duration:
            to_datetime = self.work_intervals[c_day][i + 1].start + timedelta(
                seconds=(pending_duration - total_duration)
            )
            return (to_datetime - from_datetime).total_seconds()
        else:
            return (self.work_intervals[c_day][i].end - from_datetime).total_seconds()

    def print_calendar_info(self):
        print("Calendar ID: %s" % self.calendar_id)
        print("Total Weekly Work: %.2f Hours" % (self.total_weekly_work / 3600))
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                print(int_week_days[i])
                for interval in self.work_intervals[i]:
                    print(
                        "    from %02d:%02d - to %02d:%02d"
                        % (
                            interval.start.hour,
                            interval.start.minute,
                            interval.end.hour,
                            interval.end.minute,
                        )
                    )

    
def get_last_available_timestamp(start: pd.Timestamp, end: pd.Timestamp, schedule: RCalendar) -> pd.Timestamp:
    """
    Get the timestamp [last_available] within the interval from [start] to [end] (i.e. [start] <= [last_available] <= [end]) such that
    the interval from [last_available] to [end] is the largest and all of it is in the working hours in the calendar [schedule].

    For example, for [start] = 09:30, [end] = 14:00, and a [schedule] of every week day from 06:00 to 09:00, and from 10:00 to 16:00. The
    [last_available] would be 10:00.

    :param start:       start of the interval where to search for the point since when all time is part of the working schedule.
    :param end:         end of the interval where to search for the point since when all time is part of the working schedule.
    :param schedule:    RCalendar with the weekly working schedule.

    :return: The earliest point within the interval from [start] to [end] since which all the time is part of working hours defined in the
             resource calendar [schedule].
    """
    # Get the latest working period previous to the end of the interval
    last_available = end
    found = False
    while not found:
        day_intervals = schedule.work_intervals[last_available.weekday()]
        for interval in reversed(day_intervals):
            # Move interval to current day
            interval_start = interval.start.replace(
                day=last_available.day,
                month=last_available.month,
                year=last_available.year,
                tzinfo=pytz.timezone("UTC"),
            )
            interval_end = interval.end.replace(
                day=last_available.day,
                month=last_available.month,
                year=last_available.year,
                tzinfo=pytz.timezone("UTC"),
            )
            if interval_end < last_available:
                # The last available is later than the end of the current working interval
                if (last_available - interval_end) > pd.Timedelta(seconds=2):
                    # Non-working time gap previous to last_available, search finished
                    found = True
                    # Correct jump to previous day if needed
                    if (
                        last_available.hour == 23
                        and last_available.minute == 59
                        and last_available.second == 59
                        and last_available.microsecond == 999999
                    ):
                        last_available = last_available + pd.Timedelta(microseconds=1)
                else:
                    # No non-working time gap, move to the start of this working interval and continue
                    last_available = last_available.replace(
                        hour=interval_start.hour,
                        minute=interval_start.minute,
                        second=interval_start.second,
                        microsecond=interval_start.microsecond,
                    )
            elif interval_start <= last_available <= interval_end:
                # The last available timestamp is within the current interval
                last_available = last_available.replace(
                    hour=interval_start.hour,
                    minute=interval_start.minute,
                    second=interval_start.second,
                    microsecond=interval_start.microsecond,
                )
        if not found:
            start_of_day = last_available.replace(hour=00, minute=00, second=00, microsecond=0)
            if (last_available - start_of_day) > pd.Timedelta(seconds=2):
                # Non-working interval between last_available and the start of the day
                found = True
            else:
                # Move to previous day at 23:59:59.999999
                last_available = (last_available - pd.Timedelta(days=1)).replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
        # If last_available moved previously to the start of the queried interval
        if last_available <= start:
            # Stop and set to the start of the queried interval
            found = True
            last_available = start
    # Return last available timestamp
    return last_available


def read_csv_log(
    log_path,
    log_ids: EventLogIDs,
    missing_resource: Optional[str] = "NOT_SET",
    sort=True,
    time_format_inference=False,
) -> pd.DataFrame:
    """
    Read an event log from a CSV file given the column IDs in [log_ids]. Set the enabled_time, start_time, and end_time columns to date,
    set the NA resource cells to [missing_value] if not None, and sort by [end, start, enabled].

    :param log_path: path to the CSV log file.
    :param log_ids: IDs of the columns of the event log.
    :param missing_resource: string to set as NA value for the resource column (not set if None).
    :param sort: if true, sort event log by start, end, enabled (if available).

    :return: the read event log,
    """
    # Read log
    event_log = pd.read_csv(log_path)
    # Set case id as object
    event_log = event_log.astype({log_ids.case: object})
    # Fix missing resources (don't do it if [missing_resources] is set to None)
    if missing_resource:
        if log_ids.resource not in event_log.columns:
            event_log[log_ids.resource] = missing_resource
        else:
            event_log[log_ids.resource] = event_log[log_ids.resource].fillna(missing_resource)
    # Set resource type to string if numeric
    if log_ids.resource in event_log.columns:
        event_log[log_ids.resource] = event_log[log_ids.resource].apply(str)
    if time_format_inference:
        event_log[log_ids.end_time] = pd.to_datetime(
            event_log[log_ids.end_time], utc=True, format='mixed')
        if log_ids.start_time in event_log.columns:
            event_log[log_ids.start_time] = pd.to_datetime(
                event_log[log_ids.start_time], utc=True, format='mixed')
        if log_ids.enabled_time in event_log.columns:
            event_log[log_ids.enabled_time] = pd.to_datetime(
                event_log[log_ids.enabled_time], utc=True, format='mixed')
    else:
        # Convert timestamp value to pd.Timestamp (setting timezone to UTC)
        event_log[log_ids.end_time] = pd.to_datetime(
            event_log[log_ids.end_time], utc=True)
        if log_ids.start_time in event_log.columns:
            event_log[log_ids.start_time] = pd.to_datetime(
                event_log[log_ids.start_time], utc=True)
        if log_ids.enabled_time in event_log.columns:
            event_log[log_ids.enabled_time] = pd.to_datetime(
                event_log[log_ids.enabled_time], utc=True)
    # Sort by end time
    if sort:
        if log_ids.start_time in event_log.columns and log_ids.enabled_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time, log_ids.enabled_time])
        elif log_ids.start_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
        else:
            event_log = event_log.sort_values(log_ids.end_time)
    # Return parsed event log
    return event_log