import random
from datetime import timedelta
from django.utils import timezone
from core.models import SurveyResponse
from faker import Faker
fake = Faker()
responses = SurveyResponse.objects.all()

for r in responses:
    # Random date within last 60 days
    random_days = random.randint(0, 60)
    random_seconds = random.randint(0, 86400)

    r.created_at = fake.date_time_between(start_date='-2M', end_date='now', tzinfo=timezone.get_current_timezone())
    r.save(update_fields=["created_at"])

print("Dates randomized successfully")