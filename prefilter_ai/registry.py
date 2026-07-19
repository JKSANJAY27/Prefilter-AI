"""
registry.py — Domain Schema Registry for Prefilter AI.

Allows definition and registration of search domains, fields, datatypes,
and constraint importance weights. Supports dynamic extension for plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Importance(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class DataType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    ARRAY = "array"
    BOOLEAN = "boolean"


@dataclass
class FieldDefinition:
    name: str
    data_type: DataType
    importance: Importance = Importance.MEDIUM
    description: str = ""


@dataclass
class DomainSchema:
    name: str
    fields: dict[str, FieldDefinition] = field(default_factory=dict)
    description: str = ""

    def add_field(
        self,
        name: str,
        data_type: DataType | str,
        importance: Importance | int = Importance.MEDIUM,
        description: str = "",
    ) -> DomainSchema:
        dt = DataType(data_type)
        imp = Importance(importance)
        self.fields[name] = FieldDefinition(
            name=name, data_type=dt, importance=imp, description=description
        )
        return self


class SchemaRegistry:
    _instance: SchemaRegistry | None = None
    _schemas: dict[str, DomainSchema]

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._schemas = {}
            cls._instance._register_defaults()
        return cls._instance

    def register(self, schema: DomainSchema) -> None:
        """Register a new domain schema dynamically (supports plugin extension)."""
        self._schemas[schema.name] = schema

    def get(self, domain_name: str) -> DomainSchema | None:
        """Retrieve a domain schema definition by name."""
        return self._schemas.get(domain_name)

    def list_domains(self) -> list[str]:
        """List all currently registered domain names."""
        return list(self._schemas.keys())

    def _register_defaults(self) -> None:
        """Register the 10 out-of-the-box domain schemas."""
        # 1. Ecommerce
        ecommerce = DomainSchema("ecommerce", description="E-commerce shopping domain")
        ecommerce.add_field("product", DataType.STRING, Importance.HIGH, "Product category or name")
        ecommerce.add_field("brand", DataType.STRING, Importance.MEDIUM, "Manufacturer or brand")
        ecommerce.add_field("price", DataType.NUMBER, Importance.HIGH, "Product price constraint")
        ecommerce.add_field("color", DataType.ARRAY, Importance.LOW, "Colors or color exclusions")
        ecommerce.add_field("rating", DataType.NUMBER, Importance.LOW, "Minimum review rating")
        ecommerce.add_field("feature", DataType.STRING, Importance.MEDIUM, "Specific product feature or spec")
        self.register(ecommerce)

        # 2. Flights
        flights = DomainSchema("flights", description="Flight bookings")
        flights.add_field("origin", DataType.STRING, Importance.HIGH, "Origin airport/city")
        flights.add_field("destination", DataType.STRING, Importance.HIGH, "Destination airport/city")
        flights.add_field("cabin_class", DataType.STRING, Importance.MEDIUM, "Cabin class: business, economy, first")
        flights.add_field("stops", DataType.NUMBER, Importance.MEDIUM, "Maximum number of layovers")
        flights.add_field("price", DataType.NUMBER, Importance.HIGH, "Ticket price limit")
        self.register(flights)

        # 3. Real Estate
        real_estate = DomainSchema("real_estate", description="Property rentals and sales")
        real_estate.add_field("property_type", DataType.STRING, Importance.HIGH, "Apartment, house, townhome")
        real_estate.add_field("city", DataType.STRING, Importance.HIGH, "Target city/location")
        real_estate.add_field("price", DataType.NUMBER, Importance.HIGH, "Monthly rent or purchase budget")
        real_estate.add_field("bedrooms", DataType.NUMBER, Importance.MEDIUM, "Number of bedrooms")
        real_estate.add_field("bathrooms", DataType.NUMBER, Importance.MEDIUM, "Number of bathrooms")
        self.register(real_estate)

        # 4. Jobs
        jobs = DomainSchema("jobs", description="Job listings and career opportunities")
        jobs.add_field("job_title", DataType.STRING, Importance.HIGH, "Role title, e.g. software engineer")
        jobs.add_field("salary", DataType.NUMBER, Importance.HIGH, "Minimum salary requirement")
        jobs.add_field("location", DataType.STRING, Importance.MEDIUM, "City or remote preference")
        jobs.add_field("experience_level", DataType.STRING, Importance.MEDIUM, "Junior, Senior, Lead")
        self.register(jobs)

        # 5. Hotels
        hotels = DomainSchema("hotels", description="Hotel and lodging bookings")
        hotels.add_field("city", DataType.STRING, Importance.HIGH, "Target location")
        hotels.add_field("stars", DataType.NUMBER, Importance.MEDIUM, "Hotel star rating")
        hotels.add_field("price", DataType.NUMBER, Importance.HIGH, "Price per night limit")
        hotels.add_field("amenities", DataType.ARRAY, Importance.LOW, "Pool, breakfast, wifi")
        self.register(hotels)

        # 6. Cars
        cars = DomainSchema("cars", description="Car purchases and rentals")
        cars.add_field("make", DataType.STRING, Importance.HIGH, "Manufacturer brand")
        cars.add_field("model", DataType.STRING, Importance.HIGH, "Car model name")
        cars.add_field("body_type", DataType.STRING, Importance.MEDIUM, "SUV, sedan, truck")
        cars.add_field("price", DataType.NUMBER, Importance.HIGH, "Car purchase budget")
        cars.add_field("fuel_type", DataType.STRING, Importance.MEDIUM, "Electric, hybrid, gas")
        self.register(cars)

        # 7. Restaurants
        restaurants = DomainSchema("restaurants", description="Dine-out and delivery bookings")
        restaurants.add_field("cuisine", DataType.STRING, Importance.HIGH, "Italian, Vegan, Chinese")
        restaurants.add_field("city", DataType.STRING, Importance.HIGH, "Target location")
        restaurants.add_field("price", DataType.NUMBER, Importance.MEDIUM, "Budget per person indicator")
        restaurants.add_field("features", DataType.ARRAY, Importance.LOW, "Outdoor seating, dog friendly")
        self.register(restaurants)

        # 8. Movies
        movies = DomainSchema("movies", description="Movies and TV shows search")
        movies.add_field("genre", DataType.STRING, Importance.HIGH, "Thriller, Comedy, Drama")
        movies.add_field("platform", DataType.STRING, Importance.MEDIUM, "Netflix, Prime, Hulu")
        movies.add_field("rating", DataType.NUMBER, Importance.MEDIUM, "IMDB rating constraint")
        self.register(movies)

        # 9. Healthcare
        healthcare = DomainSchema("healthcare", description="Medical provider directory")
        healthcare.add_field("specialty", DataType.STRING, Importance.HIGH, "Therapist, Dentist, Cardiologist")
        healthcare.add_field("city", DataType.STRING, Importance.HIGH, "Target location")
        healthcare.add_field("insurance", DataType.STRING, Importance.HIGH, "Accepted insurance network")
        self.register(healthcare)

        # 10. Courses
        courses = DomainSchema("courses", description="Online courses and learning platforms")
        courses.add_field("subject", DataType.STRING, Importance.HIGH, "Python, ML, Web Dev")
        courses.add_field("price", DataType.NUMBER, Importance.HIGH, "Max budget limit")
        courses.add_field("difficulty", DataType.STRING, Importance.MEDIUM, "Beginner, Advanced")
        self.register(courses)
