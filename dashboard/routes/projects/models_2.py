from __future__ import annotations


class FilmImageInspectRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/1"] = Field(alias="schema")
    kind: Literal["film_image_inspect"]
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)
    binding_id: str | None = Field(
        pattern=r"^imgbind_[0-9a-f]{32}$",
        min_length=40,
        max_length=40,
    )
    film_reference_id: str = Field(
        pattern=r"^filmref_[0-9a-f]{32}$",
        min_length=40,
        max_length=40,
    )

    @model_validator(mode="after")
    def require_inspect_target(self) -> "FilmImageInspectRequest":
        if self.binding_id is None and self.film_reference_id is None:
            raise ValueError("an Image binding or Film reference is required")
        return self


class ImageSourceCatalogRequest(_ProjectRequest):
    schema_: Literal["studio.image-source-catalog-request/1"] = Field(alias="schema")
    kind: Literal["image_source_catalog"]


project_router = APIRouter(route_class=FiniteValidationRoute)
