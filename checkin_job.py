from ella.tasks.checkin import weekly_check_in

if __name__ == "__main__":
    # In a full deployment, Celery Beat would trigger this on a schedule.
    # For now, we can manually trigger a test against known chat IDs.
    print("Run `celery -A ella.tasks.celery_app beat` to schedule this long term.")
